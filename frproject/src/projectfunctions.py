import numpy as np
from copy import copy
import rbdl

pi = np.pi
cos = np.cos
sin = np.sin
pi = np.pi

def limitJoints(q):
    q[1] = max(min(q[1], 0.5), 0.13)
    q[2] = max(min(q[2], 2.15), -2.15)
    q[7] = max(min(q[7], 0.15), -0.08)


def dh(d, theta, a, alpha):
    """
    Calcular la matriz de transformacion homogenea asociada con los parametros
    de Denavit-Hartenberg.
    Los valores d, theta, a, alpha son escalares.

    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                  [sth,  ca*cth, -sa*cth, a*sth],
                  [0.0,      sa,      ca,     d],
                  [0.0,     0.0,     0.0,   1.0]])
    return T


def fkine_elbry420(q):
    """
    Calcular la cinematica directa del robot dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6, q7, q8]

    """
    # Longitudes (en metros)

    # Matrices DH (completar)
    #print(q)
    T1 = dh(1.08725, q[0]+np.pi/2, 0, np.pi/2)
    T2 = dh(q[1]+0.2835, np.pi, 0.115, 0)
    T3 = dh(-0.13225, q[2], 0.656, 0)
    T4 = dh(0.1565, q[3], 0.531, 0)
    T5 = dh(0, q[4]+np.pi/2, 0, np.pi/2)
    T6 = dh(0.1605, q[5]+np.pi, 0, np.pi/2)
    T7 = dh(0, q[6], 0, 0)
    T8 = dh(q[7]+0.525, 0, 0, 0)
    # Efector final con respecto a la base
    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ T8 
    return T

def ikine_newton_elbry420(xdes, q0):
    """
    Calcular la cinematica inversa de numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001

    q  = copy(q0)
     # Almacenamiento del error
    ee = []
   
    # Bucle principal
    for i in range(max_iter):
        limitJoints(q)
        J = jacobian_position(q)
        f = fkine_elbry420(q)[0:3,3]
        #print(J)
        # Error
        e = xdes-f
        # Actualización de q (método de Newton)
        q = q + np.dot(np.linalg.pinv(J), e)

        # Norma del error
        enorm = np.linalg.norm(e)
        print("Error en la iteración {}: {}".format(i, np.round(enorm,4)))
        ee.append(enorm)    # Almacena los errores
       
        # Condición de término
        if (np.linalg.norm(e) < epsilon):
            break
        if (i==max_iter-1):
            print("El algoritmo no llegó al valor deseado")

    return q

def ikine_gradient_elbry420(xdes, q0):
    """
    Calcular la cinematica inversa numericamente a partir de la configuracion articular inicial de q0.
    Emplear el metodo gradiente
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    alfa = 0.2
    q  = copy(q0)
    #Almacenamiento del error
    ee = []
   
    # Bucle principal
    for i in range(max_iter):
        limitJoints(q)
        J = jacobian_position(q)
        #print(q)
        f = fkine_elbry420(q)[0:3,3]
        # Error
        e = xdes-f
        # Actualización de q
        q = q + alfa*np.dot(J.T, e)

        # Norma del error
        enorm = np.linalg.norm(e)
        print("Error en la iteración {}: {}".format(i, np.round(enorm,4)))
        ee.append(enorm)    # Almacena los errores
       
        # Condición de término
        if (np.linalg.norm(e) < epsilon):
            print("Cinemática inversa: solución obtenida")
            break
        if (i==max_iter-1):
            print("No se llegó a la solución deseada: modificar el valor de alfa")
    return q



def jacobian_position(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    # Alocacion de memoria
    J = np.zeros((3,8))
    # Transformacion homogenea inicial (usando q)
    T = fkine_elbry420(q)
    # Iteracion para la derivada de cada columna
    for i in range(8):
        # Copiar la configuracion articular inicial (usar este dq para cada
        # incremento en una articulacion)
        dq = copy(q)
        T = fkine_elbry420(dq)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+dq)
        T_inc = fkine_elbry420(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        if (i==1 or i==7):
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])*(0.05)/delta
        else:
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])/delta
	
    return J



'''
def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x8 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6, q7, q8]

    """
    J = np.zeros((7,8))
    # Implementar este Jacobiano aqui
    T = fkine_elbry420(q)
    Q = rot2quat(T)
    for i in range(8):
    	dq = copy(q)
    	T = fkine_elbry420(dq)
    	Q = rot2quat(T)
    	dq[i] = dq[i] + delta
    	T_inc = fkine_elbry420(dq)
    	Q_inc = rot2quat(T_inc)
    	
    	if (i==1 or i==7):
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])*(0.05)/delta
        else:
            J[0:3,i]=(T_inc[0:3, 3]-T[0:3, 3])/delta
        J[3:7,i]=(Q_inc[0:4]-Q[0:4]).T/delta
    return J

'''


class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('/home/farid/project_ws/src/universal_robot/kuka_kr20_description/urdf/kr20.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq

def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]
    #print(R[0,0]+R[1,1]+R[2,2]+1.0)
    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
      T -- A homogeneous transformation
    Output:
      X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
           is Cartesian coordinates and the last part is a quaternion
    """
    #print(T)
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]] 
    
    return np.array(res)

def sVectorFromSkew(S):
    return sp.Matrix([S[2,1],S[0,2],S[1,0]])

def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R
    
def is_singular(matrix):
    det = np.linalg.det(matrix.T @ matrix)
    if np.isclose(det, 0):
        return True
    else:
        return False
        
def roty(ang):
    Ry = np.array([[cos(ang), 0, sin(ang)],
                   [0, 1, 0],
                   [-sin(ang), 0, cos(ang)]])
    return Ry

def rotz(ang):
    Rz = np.array([[cos(ang), -sin(ang), 0],
                   [sin(ang), cos(ang), 0],
                   [0,0,1]])
    return Rz

def rotx(ang):
    Rx = np.array([[1, 0, 0],
                   [0, cos(ang), -sin(ang)],
                   [0, sin(ang), cos(ang)]])
    return Rx
