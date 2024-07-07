#!/usr/bin/env python3


import rbdl
import numpy as np

if __name__ == '__main__':

  # Lectura del modelo del robot a partir de URDF (parsing)
  modelo = rbdl.loadModel('/home/farid/project_ws/src/universal_robot/kuka_kr20_description/urdf/kr20.urdf')
  # Grados de libertad
  ndof = modelo.q_size

  # Configuracion articular
  q = np.array([0.5, 0.2, 0.3, 0.8, 0.5, 0.6, 0.4, 0.8])
  # Velocidad articular
  dq = np.array([0.8, 0.7, 0.8, 0.6, 0.9, 1.0, 0.2, 0.5])
  # Aceleracion articular
  ddq = np.array([0.2, 0.5, 0.4, 0.3, 1.0, 0.5, 0.9, 0.1])
  
  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(ndof)               # Vector identidad
  
  # Torque dada la configuracion del robot
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
  
  
  # Parte 1: Calcular vector de gravedad, vector de Coriolis/centrifuga,
  # y matriz M usando solamente InverseDynamics
  
  # Calculo del vector de gravedad
  rbdl.InverseDynamics(modelo, q, zeros, zeros, tau)
  g = tau.copy()
  print("Vector de gravedad g:\n", np.round(g, 3))
  
  # Calculo del vector de Coriolis y centrífuga
  rbdl.InverseDynamics(modelo, q, dq, zeros, tau)
  c = tau - g
  print("Vector de Coriolis y centrífuga c:\n", np.round(c))
  
  # Calculo de la matriz de inercia
  for i in range(ndof):
    ei = np.ascontiguousarray(e[:,i])
    rbdl.InverseDynamics(modelo, q, zeros, ei, tau)
    M[:, i] = tau - g
  
  print("Matriz de inercia M:\n", np.round(M, 3))


  # Parte 2: Calcular M y los efectos no lineales b usando las funciones
  # CompositeRigidBodyAlgorithm y NonlinearEffects. Almacenar los resultados
  # en los arreglos llamados M2 y b2
  b2 = np.zeros(ndof)          # Para efectos no lineales
  M2 = np.zeros([ndof, ndof])  # Para matriz de inercia
  
  rbdl.CompositeRigidBodyAlgorithm(modelo, q, M2)
  print("Matriz de inercia M2:\n", np.round(M2, 3))
  
  rbdl.NonlinearEffects(modelo, q, dq, b2)
  print("Matriz b:\n", np.round(b2, 3)) 
  
  
  # Parte 2: Verificacion de valores

  
  

  # Parte 3: Verificacion de la expresion de la dinamicap
  rbdl.InverseDynamics(modelo, q, dq, ddq, tau)
  
  print("Tau: \n", np.round(tau, 3))
  
  print("Método 1: \n", np.round((M.dot(ddq) + c + g), 3))
  
  print("Método 2:\n", np.round((M2.dot(ddq) + b2), 3))
  
