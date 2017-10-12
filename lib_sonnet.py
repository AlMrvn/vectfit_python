##### sonnet utilitaries
from numpy import *

def read_SonnetOutput(fname):
   """reads SONNET 'output' file. return the name of file.txt, the name of parameter and parameters"""
   s = open(fname+str('output_files.txt'),'r').read()
   #s_header = s[:281]
   s = s.split('\n\n')
   s.pop(0)
   s.pop(0)
   s.pop(0)
   s.pop(-1)
   name = []
   name_params = []
   params = []
   for p in (s[0].split('\n')[2:-1]):
      name_params.append((p[:p.find('=')]))
   for y in s:
      name.append(y.split('\n')[0][12:])
      for p in y.split('\n')[2:-1]:
         params.append((float(p[p.find('=')+1:])))
   params = array(params).reshape((len(s),len(name_params)))
   return name,name_params,params

def extractData(fname):
   """ extract the Sij (Yij, Zij...) matrix from sonnet .txt file"""
   s = open(fname, 'r').read()
   index_header = s.find('[Network Data]')    # len = 14
   s_header =s[:index_header+14]
   s = s[index_header+15:]
   #Find the number of port
   number_of_port = int((s_header[s_header.find('[Number of Ports]'):].split('\n',1)[0]).split(' ')[-1])
   x = array(s.split(), dtype=double)
   first_col = x[::2*number_of_port**2+1]
   tables = x[(arange(x.size) % (2*number_of_port**2+1)) > 0]
   tables = tables.reshape(-1, number_of_port , 2*number_of_port)
   tables_c = zeros((len(first_col), number_of_port,number_of_port)).astype(complex)
   for k in range(number_of_port):
      for j in range(number_of_port):
        tables_c[:,k,j] = tables[:,k,2*j]+1j*tables[:,k,2*j+1]
   return first_col,tables_c
 
def S_from_Y(fname,number_of_port,Zc=50.):
   freq,Y = extractData(fname)
   Nf = len(freq)
   A = Zc*Y[:,:number_of_port,:number_of_port]
   B = Zc*Y[:,:number_of_port,number_of_port:]   
   C = Zc*Y[:,number_of_port:,:number_of_port]   
   D = Zc*Y[:,number_of_port:,number_of_port:]
   S = zeros((Nf,number_of_port,number_of_port))
   S = S.astype(complex)
   for i in range(Nf):
      invAp = linalg.inv(diag(ones(number_of_port)) + A[i,:,:])
      Am = diag(ones(number_of_port)) - A[i,:,:]
      schur = linalg.inv(D[i,:,:] - (C[i,:,:].dot(invAp)).dot(B[i,:,:]))
      S[i,:,:] = Am.dot(invAp) + ((((Am.dot(invAp) + diag(ones(number_of_port))).dot(B[i,:,:])).dot(schur)).dot(C[i,:,:])).dot(invAp)
   return freq,S

def y(freq,Y):
   """ return the matrix of dipole """
   Nf = len(freq)
   Nd = shape(Y)[-1]
   y = - Y
   for i in range(Nd):
      y[:,i,i] += Y[:,i,i]+sum(Y[:,i,:],axis=1)
   return freq,y

def func(freq,Y):
    Nf = len(freq)
    Nd = shape(Y)[-1]
    n = sum([x for x in range(1,Nd+1)])
    f = zeros((Nf,n), dtype = complex64)
    for i in range(Nd):
        for j in range(i,Nd):
            f[:,j + sum([x for x in range(1,i)]) ] = Y[:,i,j]
    return f