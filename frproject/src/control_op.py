#!/usr/bin/env python3

import rospy
import os
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from markers import *
from projectfunctions import *
from roslib import packages

import rbdl

if __name__ == '__main__':

  rospy.init_node("control_pdg")
  pub = rospy.Publisher('joint_states', JointState, queue_size=1000)
  bmarker_actual  = BallMarker(color['RED'])
  bmarker_deseado = BallMarker(color['GREEN'])
  # Archivos donde se almacenara los datos
  fqact = open("/tmp/qactual.txt", "w")
  fqdes = open("/tmp/qdeseado.txt", "w")
  fxact = open("/tmp/xactual.txt", "w")
  fxdes = open("/tmp/xdeseado.txt", "w")
 
  # Nombres de las articulaciones
  jnames = ['q1','q2','q3','q4','q5','q6','q7','q8']
  # Objeto (mensaje) de tipo JointState
  jstate = JointState()
  # Valores del mensaje
  jstate.header.stamp = rospy.Time.now()
  jstate.name = jnames
 
  # =============================================================
  # Configuracion articular inicial (en radianes)
  q = np.array([0.0, -0.15, 1.7, -2.2, -1.6, 0.0, 0.01, 0.05])
  # Velocidad inicial
  dq = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
  # Posición deseada
  xdes = np.array([0.8, 1.2, 0.8])

  qdes = ikine_newton_elbry420(xdes, q)
  # =============================================================
 
  # Posicion resultante de la configuracion articular deseada
  xdes = fkine_elbry420(qdes)[0:3,3]
  # Copiar la configuracion articular en el mensaje a ser publicado
  jstate.position = q
  pub.publish(jstate)
 
  # Modelo RBDL
  modelo = rbdl.loadModel('/home/farid/project_ws/src/universal_robot/kuka_kr20_description/urdf/kr20.urdf')
  ndof = modelo.q_size     # Grados de libertad
 
  # Frecuencia del envio (en Hz)
  freq = 200
  dt = 1.0/freq
  rate = rospy.Rate(freq)
 
  # Simulador dinamico del robot
  robot = Robot(q, dq, ndof, dt)

  # Se definen las ganancias del controlador, diagonales

  Kp = 70*np.diag(np.array([50, 250, 150]))
  Kd = 30*np.diag(np.array([100, 100, 10, 10, 10, 100, 10, 20]))

  # Arrays numpy
  zeros = np.zeros(ndof)          # Vector de ceros
  tau   = np.zeros(ndof)          # Para torque
  g     = np.zeros(ndof)          # Para la gravedad
  c     = np.zeros(ndof)          # Para el vector de Coriolis+centrifuga
  M     = np.zeros([ndof, ndof])  # Para la matriz de inercia
  e     = np.eye(ndof)            # Vector identidad
  
  i = 0
  # Bucle de ejecucion continua
  t = 0.0
  while not rospy.is_shutdown():
    i+=1
    # Leer valores del simulador
    q  = robot.read_joint_positions()
    dq = robot.read_joint_velocities()
    # Posicion actual del efector final
    limitJoints(q)   
    x = fkine_elbry420(q)[0:3,3]
    # Tiempo actual (necesario como indicador para ROS)
    jstate.header.stamp = rospy.Time.now()

    # Almacenamiento de datos
    fxact.write(str(t)+' '+str(x[0])+' '+str(x[1])+' '+str(x[2])+'\n')
    fxdes.write(str(t)+' '+str(xdes[0])+' '+str(xdes[1])+' '+str(xdes[2])+'\n')
    fqact.write(str(t)+' '+str(q[0])+' '+str(q[1])+' '+ str(q[2])+' '+ str(q[3])+' '+str(q[4])+' '+str(q[5])+' '+str(q[6])+' '+str(q[7])+'\n ')
    fqdes.write(str(t)+' '+str(qdes[0])+' '+str(qdes[1])+' '+ str(qdes[2])+' '+ str(qdes[3])+' '+str(qdes[4])+' '+str(qdes[5])+' '+str(qdes[6])+' '+str(qdes[7])+'\n ')

    # ----------------------------
    # Control dinamico 
    # ----------------------------
    rbdl.InverseDynamics(modelo, q, zeros, zeros, g)
    Ja = jacobian_position(q)
    u =g+ Ja.T @ Kp.dot(np.subtract(xdes, x)) - Kd.dot(dq)  #Ley de control
    print(u)
   
    # Simulacion del robot
    robot.send_command(u)

    # Publicacion del mensaje
    jstate.position = q
    pub.publish(jstate)
    bmarker_deseado.xyz(xdes)
    bmarker_actual.xyz(x)
    t = t+dt
    # Esperar hasta la siguiente  iteracion
    rate.sleep()

  fqact.close()
  fqdes.close()
  fxact.close()
  fxdes.close()

#--------------------------------------------------------------------------------------------------  
  # Leer data de los archivos
  qcurrent_data = np.loadtxt("/tmp/qactual.txt")
  qdesired_data = np.loadtxt("/tmp/qdeseado.txt")
  # Genera vector de tiempo de paso constante
  num_samples = qcurrent_data.shape[0]
  time = np.linspace(0, num_samples / freq, num_samples)

  # Plotea
  plt.figure(figsize=(10, 8))

  plt.subplot(3, 3, 1)
  plt.plot(time, qcurrent_data[:, 1], label='Actual')
  plt.plot(time, qdesired_data[:, 1], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q1 vs t')

  plt.subplot(3, 3, 2)
  plt.plot(time, qcurrent_data[:, 2], label='Actual')
  plt.plot(time, qdesired_data[:, 2], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q2 vs t')

  plt.subplot(3, 3, 3)
  plt.plot(time, qcurrent_data[:, 3], label='Actual')
  plt.plot(time, qdesired_data[:, 3], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q3 vs t')
 
  plt.subplot(3, 3, 4)
  plt.plot(time, qcurrent_data[:, 4], label='Actual')
  plt.plot(time, qdesired_data[:, 4], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q4 vs t')
   
  plt.subplot(3, 3, 5)
  plt.plot(time, qcurrent_data[:, 5], label='Actual')
  plt.plot(time, qdesired_data[:, 5], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q5 vs t')    
 
  plt.subplot(3, 3, 6)
  plt.plot(time, qcurrent_data[:, 6], label='Actual')
  plt.plot(time, qdesired_data[:, 6], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q6 vs t')
  
  plt.subplot(3, 3, 7)
  plt.plot(time, qcurrent_data[:, 7], label='Actual')
  plt.plot(time, qdesired_data[:, 7], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q7 vs t')

  plt.subplot(3, 3, 8)
  plt.plot(time, qcurrent_data[:, 8], label='Actual')
  plt.plot(time, qdesired_data[:, 8], label='Desired', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Angle [rad]')
  plt.legend()
  plt.title('q8 vs t')

  plt.tight_layout()

#----------------------------------------------------------------------------------------
  # Leer data de los archivos
  xcurrent_data = np.loadtxt("/tmp/xactual.txt")
  xdesired_data = np.loadtxt("/tmp/xdeseado.txt")
  # Genera vector de tiempo de paso constante
  num_samples = xcurrent_data.shape[0]
  time = np.linspace(0, num_samples / freq, num_samples)

  # Plotea
  plt.figure(figsize=(10, 8))

  plt.subplot(3, 1, 1)
  plt.plot(time, xcurrent_data[:, 1], label='Actual X')
  plt.plot(time, xdesired_data[:, 1], label='Desired X', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('X Position [m]')
  plt.legend()

  plt.subplot(3, 1, 2)
  plt.plot(time, xcurrent_data[:, 2], label='Actual Y')
  plt.plot(time, xdesired_data[:, 2], label='Desired Y', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Y Position [m]')
  plt.legend()

  plt.subplot(3, 1, 3)
  plt.plot(time, xcurrent_data[:, 3], label='Actual Z')
  plt.plot(time, xdesired_data[:, 3], label='Desired Z', linestyle='--')
  plt.xlabel('Time [s]')
  plt.ylabel('Z Position [m]')
  plt.legend()

  plt.tight_layout()
  plt.show()

