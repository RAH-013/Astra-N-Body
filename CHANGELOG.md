# Changelog

## 0.2.2

- La validación CPU/OpenCL comprueba errores relativos de velocidad además de posición y energía.
- La suite OpenCL prueba FP32 y FP64 cuando el dispositivo anuncia soporte FP64.
- Se actualizó la versión del proyecto y la documentación.


## 0.2.1

- Se incorporó exportación CSV del estado final y de snapshots temporales.
- Se añadieron herramientas Python para visualizar snapshots y evolución 3D.
- Se añadió un benchmark repetible con calentamiento OpenCL, medianas y gráficas.
- Se separó la configuración Docker portable CPU de la configuración GPU para Linux.
- Se hizo explícita la selección de dispositivos GPU OpenCL.
- La validación ahora reporta fallos de inicialización y distingue tiempo de kernel y tiempo end-to-end.
- Se añadieron controles de resultados finitos, conservación energética y momento.
- Se ampliaron las pruebas CPU y la documentación técnica y experimental.
- Se corrigió la acumulación de tiempos y transferencias al ejecutar simulaciones por bloques.
