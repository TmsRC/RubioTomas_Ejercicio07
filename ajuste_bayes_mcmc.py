import numpy as np
import matplotlib.pyplot as plt


datosFull = np.loadtxt('notas_andes.dat', skiprows=1)
Y = datosFull[:,4]
X = datosFull[:,:4]

Sigma = 0.1*np.ones(len(Y))


datosFull = np.loadtxt('notas_andes.dat', skiprows=1)
Y = datosFull[:,4]
X = datosFull[:,:4]

Sigma = 0.1*np.ones(len(Y))


def log_prior(betas):
    return np.log(np.prod((betas>=-2)*(betas<=2)))

def evaluar(X,betas):
    coefs = betas[1:]
    interceptos = betas[0]
    return np.matmul(X,coefs)+interceptos

def log_verosimilitud_modelo(betas,X,Y,Sigma):
    const = np.log(1/(Sigma*np.sqrt(2*np.pi)))
    #print(np.matmul(X_k,betas)[0])
    delta = evaluar(X,betas) - Y
    chi2 = (delta/Sigma)**2
    return np.sum(const-chi2/2)
    #return np.sum(-chi2/2)
    
    
N = 20000
#betas = [np.random.rand(5)-0.5]
betas = [np.zeros(5)]
#log_pos = [np.log(prior(betas[0]))+log_verosimilitud_modelo(betas[0],X,Y,Sigma)]
for i in range(0,N):
    paso = np.random.normal(loc=0.0,scale=0.05,size=len(betas[i]))
    nuevos_betas = betas[i]+paso
    log_nuevo = log_verosimilitud_modelo(nuevos_betas,X,Y,Sigma) #+ log_prior(nuevos_betas)
    log_viejo = log_verosimilitud_modelo(betas[i],X,Y,Sigma) #+ log_prior(betas[i])
    r = min(0,log_nuevo-log_viejo)
    alfa = np.random.rand()
    if np.exp(r)>=alfa:
        betas.append(nuevos_betas)
        #log_pos.append(log_nuevo)
    else:
        betas.append(betas[i])
        #log_pos.append(log_viejo)
betas = np.array(betas[10000:])


estimadores = np.mean(betas,axis=0)
desviaciones = np.std(betas,axis=0)
print(estimadores)
log_posteriores = []

for x in range(len(betas[:,0])):
    betas_x = betas[x,:]
    log_x = log_verosimilitud_modelo(betas_x,X,Y,Sigma)
    log_posteriores.append(log_x)

log_posteriores = np.array(log_posteriores)
posteriores = np.exp(log_posteriores-np.amax(log_posteriores))

fig,axii = plt.subplots(2,3,figsize=(9,6))

axes = list(axii[0])+list(axii[1])

for i in range(0,5):
    
    axes[i].hist(betas[:,i],bins=15,density=True)
    axes[i].set_title(r'$\beta_{} = {:.2f} \pm {:.2f}$ '.format(i,estimadores[i],desviaciones[i]))

fig.tight_layout()
plt.savefig('ajuste_bayes_mcmc.png')