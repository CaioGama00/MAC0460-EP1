### Exercício Programa 1 ######################################################
#  AO PREENCHER ESSE CABEÇALHO COM O MEU NOME E O MEU NÚMERO USP,             #
#  DECLARO QUE SOU O ÚNICO AUTOR E RESPONSÁVEL POR ESSE PROGRAMA.             #
#  TODAS AS PARTES ORIGINAIS DESSE EXERCÍCIO PROGRAMA (EP) FORAM              #
#  DESENVOLVIDAS E IMPLEMENTADAS POR MIM SEGUINDO AS INSTRUÇÕES               #
#  DESSE EP E QUE PORTANTO NÃO CONSTITUEM DESONESTIDADE ACADÊMICA             #
#  OU PLÁGIO.                                                                 #
#  DECLARO TAMBÉM QUE SOU RESPONSÁVEL POR TODAS AS CÓPIAS                     #
#  DESSE PROGRAMA E QUE EU NÃO DISTRIBUI OU FACILITEI A                       #
#  SUA DISTRIBUIÇÃO. ESTOU CIENTE QUE OS CASOS DE PLÁGIO E                    #
#  DESONESTIDADE ACADÊMICA SERÃO TRATADOS SEGUNDO OS CRITÉRIOS                #
#  DIVULGADOS NA PÁGINA DA DISCIPLINA.                                        #
#  ENTENDO QUE EPS SEM ASSINATURA NÃO SERÃO CORRIGIDOS E,                     #
#  AINDA ASSIM, PODERÃO SER PUNIDOS POR DESONESTIDADE ACADÊMICA.              #
#                                                                             #
#  Nome :                                                                     #
#  NUSP :                                                                     #
#  Turma:                                                                     #
#  Prof.:                                                                     #
###############################################################################
""" programa ep01.py

Simular a diferença entre erro in e erro out a partir de um Universo (pré-estabelecido:
dados X pertencente a R² e Y: {-1,+1} com N pontos)
e um conjunto de hipóteses bem definido e finito!
LEIA O ENUNCIADO COMPLETO NO ARQUIVO EM  PDF

"""
import math 
import numpy as np

def standardization(X): # valor:0.5
  ''' 
    Given a np.array X returns X_std: mean = 0, std = 1 (not inplace - pure function)
  '''
  # seu código aqui
  mean = np.mean(X)
  std = np.std(X)
  X_std = (X - mean) / std

  return X_std
  
def calc_error(Y,Y_hat): # valor:1.0
  ''' 
    Given Y (labels) and Y_hat(predicts), returns normalized error
    Inputs:
      Y: np.array or list
      Y_hat: np.array or list
  '''
  # seu código aqui
  count = 0
  n = len(Y)

  for i in range(n):
    if Y[i] != Y_hat[i]:
      count += 1

  norm_error = count/n

  return norm_error
  
def sampling(N,X,Y,random_state=42): # valor:0.5
  '''
    Given the N #of samples (to sampling), X (np.array) and Y (labels - np.array)
    returns N random samples (X,y) type: np.array
  '''
  # seu código aqui
  np.random.seed(random_state)

  X_sample = []
  Y_sample = []
  i = 0
  R = []

  while i < N:
    rand = np.random.randint(0,N) 
    while rand in R:
       rand = np.random.randint(0,N)
    X_sample.append(X[rand])
    Y_sample.append(Y[rand])
    R.append(rand)
    i += 1
  
  X_sample = np.asarray(X_sample)
  Y_sample = np.asarray(Y_sample)

  return (X_sample, Y_sample)

  
def euclidean_dist(p,q): # valor:0.5
  '''
    Given two points (np.array) returns the euclidean distance between them
  '''
  # seu código aqui
  dist = math.sqrt((q[0] - p[0])**2 + (q[1] - p[1])**2)
  return dist

def diagonais(X,M,b): # valor:2.5
  '''
    Função Diagonais: retas 45º (coeficiente angular +1 e -1  variando bias 
    um tanto para frente e um tanto para trás - passo do bias (b passado por parâmetro) 
    definido pelo intervalo [-M//4,M//4)

    Sabendo que: 
      x0 * w[0] + x1 * w[1] + bias = 0 e que
      w = [1,1] no caso da reta com inclinação negativa e
      w = [1,-1] no caso da reta com inclinação positiva

    A seguinte ordem deve ser utilizada:
      bias partindo de -M//4 até M//4 (exclusive)
      A reta com inclinação negativa (coef == -1), vetor w = [1,1] (perpendicular a reta), e bias é calculda primeiro 
      e a na sequência reta com inclinação positiva, vetor w = [1,-1], e o mesmo bias.
      Conforme mostrado nos plots!

	parâmetros:
		X: np.array
		M: número de hipóteses do universo (número inteiro) - espera-se um múltiplo de 4
	Retorna 
		predict: np.array de np.array de y_hat, um y_hat para cada hipótese (reta), deve ter tamanho M
   '''
  # seu código aqui
  assert M % 4 == 0, "M should be a multiple of 4"

  biases = np.linspace(-M//4, M//4 - 1, M)
  predicts = []

  for bias in biases:
      y_hat_negative = np.sign(X[:, 0] + X[:, 1] + bias - b)
      y_hat_positive = np.sign(X[:, 0] - X[:, 1] + bias - b)
      predicts.append(np.array([y_hat_negative, y_hat_positive]))


  return predicts
  
def egocentric(X,C,r): # valor:2.0
  '''
    Given a dataset X (np.array), C (np.array) are the points that will be used as centers, and a radius r: 
      For each point in C, Creates a circumference c, each center works as an hypothesis, and classify points inside c as +1
      otherwise -1.
      Returns all predicts (an list for each point (used as center) )
  '''
  # seu código aqui
  
  predicts = []
    
  for center in C:
      distance = np.linalg.norm(X - center, axis=1)
      inside_circumference = distance <= r
      prediction = []
      for value in inside_circumference:
          if value:
            prediction.append(1)
          else:
            prediction.append(-1)

      predicts.append(prediction)
  
  return predicts

def calc_freq(N,H_set,eps,X,Y,M=100,b=0.05,r=1,random_state = 42): # valor:3.0
  '''
  Given N # of samples(integer), H_set name of the hypotheses set 
  (string <diagonais> or <egocentric> error will be returned otherwise)
  eps: epsilon (abs(error_in - error_out) desired), X from the Universe data (np.array - complete dataset),
  Y is all label from theentire Universe(np.array), M # of hypotheses used if <diagonais> is chosen, 
  B: is the bias used when <diagonais> is chosen, r radius of the circumference if <egocentric> is chosen, 
  random_state to set the seed

  Returns:
    bound: theoretical bound for Pr[abs(error_in - error_out) > eps]
    probs: approximated probability of Pr[abs(error_in - error_out) <= eps] by the frequency 
      (# of occurancies (abs(error_in - error_out) <= eps) / # of hipotheses)
  '''
  # seu código aqui
  np.random.seed(random_state)  # Set the random seed

  error_in = calculate_error_in(X, Y, H_set, M, b, r)  # Calculate in-sample error
  error_out = calculate_error_out(H_set, eps)  # Calculate out-of-sample error

  bound = calculate_theoretical_bound(N, eps)  # Calculate theoretical bound
  probs = calculate_approximate_probability(error_in, error_out, M, eps)  # Calculate approximate probability

  return (bound,probs)

def calculate_error_in(X, Y, H_set, M, B, r):
    if H_set == 'diagonais':
        predicts = diagonais(X, M, B)  # Generate hypotheses for diagonal lines
    elif H_set == 'egocentric':
        predicts = egocentric(X, r)  # Generate hypotheses for egocentric circles
    else:
        return None
    
    error_in = np.mean(predicts != Y)  # Calculate in-sample error

    return error_in

def calculate_error_out(H_set, eps):
    if H_set == 'diagonais' or H_set == 'egocentric':
        error_out = 0.1  # Placeholder value, replace with actual calculation
    else:
        return None

    return error_out


def calculate_theoretical_bound(N, eps):
    bound = np.exp(-2 * eps**2 * N)  # Calculate theoretical bound
    return bound


def calculate_approximate_probability(error_in, error_out, M, eps):
    count = np.sum(np.abs(error_in - error_out) <= eps)  # Count occurrences where |error_in - error_out| <= eps
    probs = count / M  # Calculate approximate probability

    return probs

############## Função principal - não será avaliada. ###########################
def main():
    # você pode criar ou importar um dataset qualquer e testar as funções
    # não importe o matplotlib no moodle porque dá erro pois não tem interface gráfica
    X = standardization(np.asarray([[0,0],[1,0],[3,1],[3,2],[4,4]]))
    Y = np.asarray([-1,-1,1,1,1])
    N = 2
    X_sampled_std, Y_sampled_std = sampling(N,X,Y)
    standardization(X_sampled_std)
    predicts=diagonais(X,4,1)

    print(predicts)

    return
############## chamada da função main() ########################################
if __name__ == "__main__":
    main()