"""
==============================================================================
 Proyecto MITOS - Módulo de Seguridad (Dead Man's Switch + Shamir)
==============================================================================

Este módulo implementa el subsistema de seguridad del proyecto:

  1. DeadManSwitch
       Un Dead Man's Switch (DMS) multi-party, tolerante a fallos y
       resistente a extorsiones, basado en TOTP (RFC 6238 / RFC 4226).

       Cada "fuente" (operador humano o sistema autorizado) está dada de
       alta con DOS secretos:
         - el secreto normal, y
         - un secreto de coacción derivado determinísticamente del normal
           (concatenando "_DURESS"); su uso indica que la fuente está
           siendo extorsionada o forzada.

       Propiedades de seguridad clave:

         (a) Los secretos en texto plano NUNCA se almacenan: la clase
             guarda únicamente el SHA-256 de cada secreto. Si alguien
             obtiene un volcado de memoria, no puede recuperar las
             credenciales originales.

         (b) Un atacante que observe el tráfico o el comportamiento del
             DMS NO PUEDE distinguir criptográficamente un token normal
             de un token de coacción: ambos son cadenas TOTP de 8 dígitos
             generadas con HMAC-SHA256 y truncamiento dinámico RFC 4226.
             Sólo el sistema, que conoce ambos hashes, puede diferenciarlos.

         (c) El umbral m-de-n garantiza tolerancia a fallos: el sistema
             sólo se "dispara" si caen al menos `threshold` fuentes, o
             si se detecta una señal de coacción.

         (d) Toda mutación del estado pasa por un threading.Lock, así
             que el DMS es seguro frente a múltiples hilos de recepción.

  2. ShamirSecretSharing
       Implementación clásica de Shamir (1979): un secreto se reparte en
       `n` shares de modo que cualquier subconjunto de `threshold`
       reconstruye el secreto y cualquier subconjunto estrictamente menor
       no revela NINGUNA información sobre él (secreto perfecto de
       Shannon). Toda la aritmética vive en el cuerpo finito GF(P) con
       P = 2^127 - 1 (primo de Mersenne, M_127).

==============================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


# ============================================================================
# 1. TIPOS PÚBLICOS
# ============================================================================
class HeartbeatStatus(Enum):
    """Estado global del Dead Man's Switch.

    - SAFE:      todas las fuentes están dentro de su ventana de heartbeat.
    - WARNING:   alguna fuente cayó pero todavía estamos por debajo del
                 umbral de disparo (no hay coacción).
    - TRIGGERED: el sistema debe ejecutar sus acciones de muerte/legado.
                 Esto ocurre si (a) se detectó una señal de coacción o
                 (b) el número de fuentes caídas alcanzó `threshold`.
    """

    SAFE = "safe"
    WARNING = "warning"
    TRIGGERED = "triggered"


@dataclass
class HeartbeatSource:
    """
    Registro de una fuente autorizada para emitir heartbeats.

    Diseño criptográfico:
        SÓLO almacenamos el hash SHA-256 hex del secreto. El secreto en
        texto plano nunca persiste en la instancia; ni siquiera en
        atributos privados. Esto significa que un volcado de RAM o un
        exploit de lectura en memoria no permite recuperar la credencial.

    Atributos:
        source_id:        identificador legible (ej. "alice", "watchdog-1").
        secret_hash:      SHA-256 hex del secreto (normal o de coacción).
        last_heartbeat:   timestamp del último heartbeat válido (epoch s).
        interval_seconds: ventana máxima entre heartbeats antes de
                          considerar la fuente "caída".
        is_duress:        True si esta entrada representa el secreto de
                          coacción de su `source_id`. False para el secreto
                          normal.
    """

    source_id: str
    secret_hash: str
    last_heartbeat: float
    interval_seconds: int
    is_duress: bool = False


# ============================================================================
# 2. DEAD MAN'S SWITCH
# ============================================================================
class DeadManSwitch:
    """
    Dead Man's Switch multi-party, tolerante a fallos y resistente a coacción.

    Modelo de amenaza cubierto:
      - Compromiso individual de una fuente (m-de-n: necesitamos `threshold`
        fuentes simultáneamente caídas para disparar).
      - Coacción a una fuente (la víctima emite su token *de coacción* y el
        sistema dispara silenciosamente, sin que el atacante lo perciba).
      - Lectura de RAM del proceso (sólo verá hashes, no secretos).
      - Acceso a la red (no puede distinguir token normal vs. token de
        coacción: ambos son cadenas TOTP de 8 dígitos).

    Lo que NO cubre (fuera de alcance del MVP):
      - Atacante con root persistente que pueda parchear el binario.
      - Compromiso simultáneo de >= `threshold` fuentes Y ausencia de
        coacción detectada.
    """

    # Tamaño en dígitos del TOTP. 8 dígitos -> ~10^8 espacio de búsqueda
    # por ventana de 30 s. Suficiente para uso humano + resistencia razonable.
    _TOTP_DIGITS = 8

    # ------------------------------------------------------------------
    def __init__(
        self,
        n_parties: int = 3,
        threshold: int = 2,
        interval_seconds: int = 60,
    ) -> None:
        """
        Args:
            n_parties:        número esperado de fuentes (referencia para
                              el operador; no es un límite duro).
            threshold:        cuántas fuentes deben caer simultáneamente
                              para disparar (configuración m-de-n).
            interval_seconds: ventana de tolerancia entre heartbeats para
                              considerar a una fuente viva.

        Raises:
            ValueError: si la configuración es incoherente.
        """
        if threshold < 1:
            raise ValueError("threshold debe ser >= 1")
        if n_parties < threshold:
            raise ValueError("n_parties debe ser >= threshold")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser > 0")

        self.n_parties: int = n_parties
        self.threshold: int = threshold
        self.interval_seconds: int = interval_seconds

        # Registro de fuentes.
        #   sources[source_id]         -> HeartbeatSource normal
        #   duress_sources[source_id]  -> HeartbeatSource con is_duress=True
        # Los mantenemos en estructuras separadas para que el lookup sea
        # O(1) en receive_heartbeat sin exponer el flag al exterior.
        self.sources: Dict[str, HeartbeatSource] = {}
        self.duress_sources: Dict[str, HeartbeatSource] = {}

        # Flags de estado. Una vez disparados, son monotónicos: nunca
        # vuelven a False (no queremos que el atacante los pueda "limpiar"
        # con una llamada legítima posterior).
        self.is_dead: bool = False
        self.duress_detected: bool = False

        # Auditoría: lista append-only. En producción esto iría a un log
        # firmado y exportado off-host; para el MVP basta una lista.
        self.audit_log: List[Dict] = []

        # Toda mutación de estado se hace bajo este lock para soportar
        # heartbeats concurrentes desde distintos hilos / endpoints.
        self._lock = threading.Lock()

    # ==================================================================
    #                       REGISTRO DE FUENTES
    # ==================================================================
    def register_source(self, source_id: str, secret: str) -> None:
        """
        Registra una nueva fuente con su secreto normal y deriva su
        secreto de coacción.

        El parámetro `secret` SE USA UNA SOLA VEZ dentro de esta función
        para calcular los hashes y luego deja de referenciarse. Después
        de salir, sólo quedan en memoria los SHA-256 hex (64 chars).

        Args:
            source_id: identificador único de la fuente.
            secret:    secreto compartido en texto plano. Se borra
                       implícitamente al salir del scope (GC). El llamante
                       es responsable de no logearlo ni persistirlo.

        Notas de seguridad:
            * El secreto de coacción se deriva como SHA-256(secret + "_DURESS").
              La derivación es determinística (necesaria para que el
              operador la pueda regenerar mentalmente o en su gestor) pero
              produce un hash totalmente distinto, así que los TOTP
              resultantes son indistinguibles entre sí.
            * Nunca exponemos `secret` en `audit_log`: registramos sólo
              `source_id` y un sello de tiempo.
        """
        if not source_id:
            raise ValueError("source_id no puede estar vacío")
        if not isinstance(secret, str) or not secret:
            raise ValueError("secret debe ser una cadena no vacía")

        # Hash del secreto normal y del de coacción.
        normal_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        duress_hash = hashlib.sha256(
            (secret + "_DURESS").encode("utf-8")
        ).hexdigest()

        now = time.time()
        with self._lock:
            if source_id in self.sources:
                raise ValueError(f"source_id ya registrado: {source_id}")

            self.sources[source_id] = HeartbeatSource(
                source_id=source_id,
                secret_hash=normal_hash,
                last_heartbeat=now,
                interval_seconds=self.interval_seconds,
                is_duress=False,
            )
            self.duress_sources[source_id] = HeartbeatSource(
                source_id=source_id,
                secret_hash=duress_hash,
                last_heartbeat=now,
                interval_seconds=self.interval_seconds,
                is_duress=True,
            )
            self.audit_log.append(
                {
                    "ts": now,
                    "event": "register_source",
                    "source_id": source_id,
                }
            )

        # Borrado explícito de variables locales sensibles. Esto NO es una
        # garantía real (Python guarda strings inmutables en el heap y no
        # controlamos su reciclado) pero deja la intención clara y
        # minimiza la ventana de exposición.
        del secret, normal_hash, duress_hash

    # ==================================================================
    #                       RECEPCIÓN DE HEARTBEAT
    # ==================================================================
    def receive_heartbeat(self, source_id: str, token: str) -> bool:
        """
        Procesa un heartbeat entrante.

        Lógica vital (signal de coacción):
            * Si `token` coincide con el TOTP esperado a partir del HASH DE
              COACCIÓN, activamos `self.duress_detected = True` de forma
              SILENCIOSA y devolvemos True, simulando un heartbeat normal.
              El emisor (y cualquiera que lo observe) ve un ACK idéntico
              al de un heartbeat válido.
            * Si coincide con el TOTP normal, actualizamos last_heartbeat
              y devolvemos True.
            * En cualquier otro caso devolvemos False (token inválido /
              fuente desconocida).

        Implementación:
            * Las comparaciones usan hmac.compare_digest para evitar
              filtraciones por timing.
            * El TOTP normal y el de coacción se calculan para el MISMO
              counter de tiempo, así que en una misma ventana de 30 s
              ambos son distintos pero deterministas para el sistema.

        Args:
            source_id: id de la fuente que emite.
            token:     cadena TOTP esperada (8 dígitos).

        Returns:
            True si el token es válido (normal o de coacción), False si no.
        """
        if not isinstance(token, str):
            return False

        with self._lock:
            source = self.sources.get(source_id)
            duress = self.duress_sources.get(source_id)
            if source is None or duress is None:
                # Fuente desconocida. No filtramos información: silencio.
                return False

            now = time.time()

            # Calculamos AMBOS TOTPs esperados ahora.
            expected_normal = self._generate_totp(source.secret_hash)
            expected_duress = self._generate_totp(duress.secret_hash)

            # Comparaciones en tiempo constante.
            normal_ok = hmac.compare_digest(token, expected_normal)
            duress_ok = hmac.compare_digest(token, expected_duress)

            # --- Coacción ---------------------------------------------
            # Prioridad alta: si el token de coacción coincide, el sistema
            # marca silenciosamente la coacción Y simula un ACK normal.
            # Aunque por bizarra colisión ambos coincidieran, la coacción
            # debe ganar (es la rama "más segura").
            if duress_ok:
                self.duress_detected = True
                # Actualizamos last_heartbeat de la fuente NORMAL para que
                # un observador externo no distinga este caso de un heartbeat
                # normal: las métricas públicas (last_heartbeat) cambian
                # exactamente igual.
                source.last_heartbeat = now
                duress.last_heartbeat = now
                # NOTA: no escribimos "duress" en el audit_log público;
                # registramos un evento neutro. El operador autorizado
                # detectará la coacción a través de `check_status()` o
                # `get_metrics()`, no a través del log accesible al
                # atacante in-process.
                self.audit_log.append(
                    {
                        "ts": now,
                        "event": "heartbeat_ok",
                        "source_id": source_id,
                    }
                )
                return True

            # --- Heartbeat normal -------------------------------------
            if normal_ok:
                source.last_heartbeat = now
                duress.last_heartbeat = now
                self.audit_log.append(
                    {
                        "ts": now,
                        "event": "heartbeat_ok",
                        "source_id": source_id,
                    }
                )
                return True

            # --- Token inválido ---------------------------------------
            self.audit_log.append(
                {
                    "ts": now,
                    "event": "heartbeat_invalid",
                    "source_id": source_id,
                }
            )
            return False

    # ==================================================================
    #                          ESTADO GLOBAL
    # ==================================================================
    def check_status(self) -> HeartbeatStatus:
        """
        Evalúa el estado actual del DMS sin mutar estado público.

        Reglas:
          1) Si `duress_detected` es True  -> TRIGGERED (sin importar
             cuántas fuentes estén vivas: la coacción siempre dispara).
          2) Cuenta cuántas fuentes están "caídas" (now - last_heartbeat
             > interval_seconds).
          3) Si caídas >= threshold        -> TRIGGERED.
          4) Si 0 < caídas < threshold     -> WARNING.
          5) Si caídas == 0                -> SAFE.

        Como efecto secundario, si determinamos TRIGGERED dejamos
        `self.is_dead = True` (transición monotónica).
        """
        with self._lock:
            if self.duress_detected:
                self.is_dead = True
                return HeartbeatStatus.TRIGGERED

            now = time.time()
            fallen = 0
            for src in self.sources.values():
                if (now - src.last_heartbeat) > src.interval_seconds:
                    fallen += 1

            if fallen >= self.threshold:
                self.is_dead = True
                return HeartbeatStatus.TRIGGERED
            if fallen > 0:
                return HeartbeatStatus.WARNING
            return HeartbeatStatus.SAFE

    # ==================================================================
    #                     TOTP (RFC 4226 + RFC 6238)
    # ==================================================================
    def _generate_totp(self, secret_hash: str, time_step: int = 30) -> str:
        """
        Genera un TOTP de `_TOTP_DIGITS` dígitos basado en HMAC-SHA256.

        Algoritmo (RFC 4226 dynamic truncation, parametrizado a SHA-256
        siguiendo RFC 6238 §1.2):

            counter = floor(time.time() / time_step)
            hmac    = HMAC_SHA256(key=secret_hash_bytes, msg=counter_be8)
            offset  = hmac[-1] & 0x0F
            bin_code = (hmac[offset]   & 0x7F) << 24 |
                       (hmac[offset+1] & 0xFF) << 16 |
                       (hmac[offset+2] & 0xFF) <<  8 |
                       (hmac[offset+3] & 0xFF)
            totp    = bin_code mod 10^digits   (con zero-padding por la izquierda)

        Notas:
            * Usamos el HEX del SHA-256 como clave HMAC: 64 bytes ASCII,
              entropía suficiente y nunca exponemos el secreto original.
            * El secret_hash que entra aquí ya es público desde el punto
              de vista del proceso (vive en memoria); la fortaleza del
              TOTP descansa en (a) que el secreto original no se filtra
              y (b) en que el espacio de la ventana es 10^8 ~ 26 bits.
            * Devolvemos string con padding de ceros para preservar la
              longitud fija (importante para comparación en tiempo
              constante).
        """
        if time_step <= 0:
            raise ValueError("time_step debe ser > 0")

        counter = int(time.time() // time_step)
        # 8 bytes big-endian = espec RFC 4226 para el contador.
        counter_bytes = struct.pack(">Q", counter)
        key = secret_hash.encode("ascii")

        digest = hmac.new(key, counter_bytes, hashlib.sha256).digest()

        # Truncamiento dinámico RFC 4226 §5.3.
        offset = digest[-1] & 0x0F
        bin_code = (
            ((digest[offset] & 0x7F) << 24)
            | ((digest[offset + 1] & 0xFF) << 16)
            | ((digest[offset + 2] & 0xFF) << 8)
            | (digest[offset + 3] & 0xFF)
        )

        modulus = 10 ** self._TOTP_DIGITS
        token_int = bin_code % modulus
        return str(token_int).zfill(self._TOTP_DIGITS)

    # ==================================================================
    #                            MÉTRICAS
    # ==================================================================
    def get_metrics(self) -> Dict:
        """
        Devuelve un snapshot del estado interno para monitoreo.

        IMPORTANTE: este dict NO contiene secretos ni hashes. Sólo
        exposiciones agregadas seguras para mostrar en un dashboard.

        Returns:
            dict con:
              - n_sources_registered: cuántas fuentes hay dadas de alta.
              - threshold:            umbral m-de-n configurado.
              - interval_seconds:     ventana de tolerancia.
              - status:               valor actual de check_status().
              - is_dead:              flag monotónico de disparo.
              - duress_detected:      flag monotónico de coacción.
              - audit_log_length:     tamaño del log (no su contenido).
              - last_audit_event:     último evento (sin secretos).
        """
        with self._lock:
            status = (
                HeartbeatStatus.TRIGGERED
                if self.duress_detected
                else self._status_unlocked()
            )
            return {
                "n_sources_registered": len(self.sources),
                "threshold": self.threshold,
                "interval_seconds": self.interval_seconds,
                "status": status.value,
                "is_dead": self.is_dead,
                "duress_detected": self.duress_detected,
                "audit_log_length": len(self.audit_log),
                "last_audit_event": (
                    self.audit_log[-1] if self.audit_log else None
                ),
            }

    def _status_unlocked(self) -> HeartbeatStatus:
        """Versión de check_status que asume el lock ya tomado."""
        now = time.time()
        fallen = sum(
            1
            for src in self.sources.values()
            if (now - src.last_heartbeat) > src.interval_seconds
        )
        if fallen >= self.threshold:
            return HeartbeatStatus.TRIGGERED
        if fallen > 0:
            return HeartbeatStatus.WARNING
        return HeartbeatStatus.SAFE


# ============================================================================
# 3. SHAMIR SECRET SHARING (m-de-n sobre GF(2^127 - 1))
# ============================================================================
class ShamirSecretSharing:
    """
    Esquema (k, n) de Shamir sobre el cuerpo finito GF(P), con
    P = 2^127 - 1 (primo de Mersenne M_127).

    Garantía de seguridad:
        * Cualquier subconjunto de >= threshold shares reconstruye el
          secreto exactamente.
        * Cualquier subconjunto de < threshold shares revela cero
          información sobre el secreto: para cada hipótesis del secreto,
          existe exactamente un polinomio compatible con los shares
          observados (secreto perfecto de Shannon).
        * Por tanto, en una configuración (n, k), un atacante que
          comprometa hasta k-1 partes no aprende nada.

    Limitaciones:
        * El secreto debe ser un entero en [0, P). Para repartir secretos
          mayores, divídelos en bloques de 16 bytes y aplica Shamir a
          cada bloque (fuera del alcance de esta clase).
        * Esta implementación es síncrona y no constant-time; se asume
          que se ejecuta en un entorno de confianza.
    """

    # 2**127 - 1 (Mersenne M_127). Es primo y permite operar con
    # secretos de hasta 127 bits sin colisiones.
    PRIME: int = 2 ** 127 - 1

    # ------------------------------------------------------------------
    @classmethod
    def split(
        cls, secret: int, n_shares: int, threshold: int
    ) -> List[Tuple[int, int]]:
        """
        Divide `secret` en `n_shares` shares con reconstrucción mínima
        de `threshold` shares.

        Polinomio:
            f(x) = secret + a_1 * x + a_2 * x^2 + ... + a_{k-1} * x^{k-1}
                                                                   (mod P)
            donde k = threshold y a_1..a_{k-1} se eligen uniformemente
            al azar en [0, P) usando secrets.randbelow (RNG criptográfico).

        Shares:
            Los puntos devueltos son (i, f(i)) para i = 1..n_shares.
            No usamos i = 0 porque f(0) == secret revelaría el secreto.

        Args:
            secret:    entero en [0, P) a repartir.
            n_shares:  cuántos shares totales generar.
            threshold: mínimo de shares para reconstruir (1 <= k <= n).

        Returns:
            Lista de tuplas (x, y) con 1 <= x <= n_shares.

        Raises:
            ValueError: si los parámetros son inconsistentes o el secreto
                        está fuera de rango.
        """
        if not isinstance(secret, int):
            raise ValueError("secret debe ser int")
        if not (0 <= secret < cls.PRIME):
            raise ValueError(f"secret debe estar en [0, {cls.PRIME})")
        if threshold < 1:
            raise ValueError("threshold debe ser >= 1")
        if n_shares < threshold:
            raise ValueError("n_shares debe ser >= threshold")
        # x va de 1..n_shares; necesitamos n_shares < PRIME para que cada
        # x sea distinto módulo P. Con P de 127 bits esto es siempre cierto
        # en la práctica, pero lo dejamos explícito.
        if n_shares >= cls.PRIME:
            raise ValueError("n_shares demasiado grande para este P")

        # Coeficientes a_1..a_{k-1} uniformes en [0, P).
        # secrets.randbelow utiliza el RNG del SO (urandom): apto para uso
        # criptográfico.
        coeffs: List[int] = [secret] + [
            secrets.randbelow(cls.PRIME) for _ in range(threshold - 1)
        ]

        shares: List[Tuple[int, int]] = []
        for x in range(1, n_shares + 1):
            # Evaluación por Horner: O(k) multiplicaciones modulares.
            #   f(x) = ((...((a_{k-1} * x + a_{k-2}) * x + a_{k-3}) ...) * x + a_0)
            y = 0
            for coeff in reversed(coeffs):
                y = (y * x + coeff) % cls.PRIME
            shares.append((x, y))

        return shares

    # ------------------------------------------------------------------
    @classmethod
    def reconstruct(cls, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruye el secreto a partir de >= threshold shares válidos
        usando interpolación de Lagrange evaluada en x = 0.

        Fórmula (con todas las operaciones mod P):

                       k                        x_j
            f(0) =   sigma   y_i  *  product  ----------
                      i=1            j != i   x_j - x_i

        Equivalentemente:

            L_i(0) = ( product_{j != i} (-x_j) ) * ( product_{j != i} (x_i - x_j) )^{-1}

        El inverso modular se calcula con pow(a, -1, PRIME) (Python 3.8+),
        que internamente usa el algoritmo extendido de Euclides.

        Args:
            shares: lista de >= 1 tuplas (x_i, y_i) distintas en x.

        Returns:
            El entero original `secret`.

        Raises:
            ValueError: si la lista está vacía, tiene x duplicados, o
                        contiene valores fuera del cuerpo.
        """
        if not shares:
            raise ValueError("shares no puede estar vacío")

        xs = [x for x, _ in shares]
        if len(set(xs)) != len(xs):
            raise ValueError("shares con x duplicados: imposible interpolar")

        for x, y in shares:
            if not isinstance(x, int) or not isinstance(y, int):
                raise ValueError("shares deben ser tuplas (int, int)")
            if not (0 <= y < cls.PRIME):
                raise ValueError("y fuera del cuerpo GF(P)")

        p = cls.PRIME
        secret_acc = 0

        for i, (xi, yi) in enumerate(shares):
            # Numerador y denominador del lagrangiano L_i(0).
            num = 1
            den = 1
            for j, (xj, _) in enumerate(shares):
                if i == j:
                    continue
                # L_i(0) factor j: (0 - xj) / (xi - xj)
                num = (num * (-xj)) % p
                den = (den * (xi - xj)) % p

            # Inverso modular del denominador en GF(P).
            # pow(a, -1, p) requiere a != 0 mod p, lo cual está garantizado
            # porque xi != xj implica (xi - xj) != 0 mod p (P es enorme).
            den_inv = pow(den, -1, p)
            lagrange_i = (num * den_inv) % p

            secret_acc = (secret_acc + yi * lagrange_i) % p

        return secret_acc
