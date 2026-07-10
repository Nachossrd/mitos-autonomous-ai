"""Subsistema de auto-modificación segura (Fase 3).

Provee tres piezas que cooperan para que el daemon pueda reescribir su
propio código sin romperse:

  - Introspector: lee y entiende el AST del proyecto.
  - CodeRewriter: aplica cambios sobre archivos reales con backup.
  - ChangeValidator: verifica que un cambio no rompe contratos públicos.
"""
