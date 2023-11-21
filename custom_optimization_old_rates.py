"""
Created on Mon Jul 24 22:41:31 2023
@author: Toluwani Okunola 
"""

#%%
import time
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.optimize import minimize
from numpy import exp
import importlib
import os
import sys
import amici
import amici.plotting
import numpy as np
from numpy import log10
import pypesto
import pypesto.optimize as optimize
import pypesto.visualize as visualize
from scipy.signal import find_peaks
from scipy import signal
from cycler import cycler
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })
#%%
# sbml file we want to import
sbml_file = "recovery_model_60Hz_no_F_F.xml"
# name of the model that will also be the name of the python module
model_name = "no_sensitivities_recovery_model_60Hz_no_F_F"
#directory to which the generated model code is written
model_output_dir = model_name
#%%

def my_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

number_of_sites = 5
n_starts = 40

animal=1

#param_combo_optimized = 'c1'
#param_combo_optimized = 'c2'
#param_combo_optimized = 'c3'
#param_combo_optimized = 'c4'
param_combo_optimized = 'c5'
#param_combo_optimized = 'c6'

objective = 'peak_values_NF_prime_and_current'
#objective = 'peak_values_NF_prime'
#objective = 'peak_values_current'
#objective = 'frechet_NF_prime'
#objective = 'frechet_current'
#objective = 'frechet_NF_prime_and_current'

#measurements = 'art'
measurements = 'real'

opt_algotrithm = 'Nelder-Mead'

start_time = time.time()

run_name = f"animal{animal}_{param_combo_optimized}_{number_of_sites}_sites_{measurements}_{n_starts}_starts_{objective}_{opt_algotrithm}_{start_time}"


my_mkdir('results_animals_bounds')


run_dir=f"results_animals_bounds/{run_name}"


my_mkdir(run_dir)
#%%
if param_combo_optimized == 'c1':
    param_indices = [0,2,3,-3]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
                                log10(5000)),(log10(1e-8),log10(5000)))
if param_combo_optimized == 'c2':
    param_indices = [0,2,3,-3,-2]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
                                log10(5000)),(log10(1e-8),log10(5000)),(log10(1),log10(100)))
if param_combo_optimized == 'c3':
    param_indices = [0,2,3,-3,-2,-1]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
                    log10(5000)),(log10(1e-8),log10(5000)),(log10(1),log10(100)),(log10(1),log10(100)))
if param_combo_optimized == 'c4':
    param_indices = [0,2,3,6,-3]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
                    log10(5000)),(log10(1e-8),log10(5000)),(log10(1e-8),log10(5000)))   
if param_combo_optimized == 'c5':
    param_indices = [0,2,3,6,-3,-2]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
                    log10(5000)),(log10(1e-8),log10(5000)),(log10(1e-8),log10(5000)),(log10(1),log10(100)))
if param_combo_optimized == 'c6':
    param_indices = [0,2,3,6,-3,-2,-1]
    param_bounds = ((log10(1),log10(1000)),(log10(1e-8),log10(5000)),(log10(1e-8),
    log10(5000)),(log10(1e-8),log10(5000)),(log10(1e-8),log10(5000)),(log10(1),log10(100)),(log10(1),log10(100)))

#%%
# sbml_reader = libsbml.SBMLReader()
# sbml_doc = sbml_reader.readSBML(sbml_file)
# sbml_model = sbml_doc.getModel()
# import sbml model, compile and generate amici module
#sbml_importer = amici.SbmlImporter(sbml_file)
#sbml_importer.sbml2amici(model_name, model_output_dir, verbose=True,generate_sensitivity_code = False)

#
# load amici module (the usual starting point later for the analysis)
#sys.path.insert(0, os.path.abspath(model_output_dir))
model_module =  amici.import_model_module(model_name,model_output_dir)
model = model_module.getModel()
ts = np.load('ts.npy')
times = ts
np.save('times.npy',times)


#%%
times = np.load('times.npy')
#print(ts,times,ts.shape,times.shape)
model.setTimepoints(times)
model.setParameterScale(amici.ParameterScaling.log10)

solver = model.getSolver()
solver.setNewtonMaxSteps(60)
solver.setMaxSteps(159999999999999999)

# Amici testrun
rdata = amici.runAmiciSimulation(model, solver,None)
amici.plotting.plotStateTrajectories(rdata)
plt.savefig('nominal_plot')

#%% 
#rdata = amici.runAmiciSimulation(model, solver)
#for key, value in rdata.items():
#     print('%12s: ' % key, value)



#%%
print("Parameter values",model.getParameters())
print("Model name:", model.getName())
print("Model parameters:", model.getParameterIds())
print("Model outputs:   ", model.getObservableIds())
print("Model states:    ", model.getStateIds())

#%%
N_var,t_wait,gV,gP,fac,sigma,L,k_base,t0,a1,mu1,a2,mu2,a3,mu3,a4,mu4,a5,mu5,a6,mu6,a7,mu7,a8,mu8,a9,mu9,a10,mu10,a11,mu11,a12,mu12,a13,mu13,a14,mu14,a15,mu15,a16,mu16,a17,mu17,a18,mu18,a19,mu19,a20,mu20,a21,mu21,a22,mu22,a23,mu23,a24,mu24,a25,mu25,a26,mu26,a27,mu27,a28,mu28,a29,mu29,a30,mu30,a31,mu31,a32,mu32,a33,mu33,a34,mu34,a35,mu35,a36,mu36,a37,mu37,a38,mu38,a39,mu39,a40,mu40,a41,mu41,a42,mu42,a43,mu43,a44,mu44,a45,mu45,a46,mu46,a47,mu47,a48,mu48,a49,mu49,a50,mu50,a51,mu51,a52,mu52,a53,mu53,a54,mu54,a55,mu55,a56,mu56,a57,mu57,a58,mu58,a59,mu59,a60,mu60,a61,mu61,a62,mu62,a63,mu63,a64,mu64,a65,mu65,a66,mu66,a67,mu67,a68,mu68,a69,mu69,a70,mu70,a71,mu71,a72,mu72,a73,mu73,a74,mu74,a75,mu75,a76,mu76,a77,mu77,a78,mu78,a79,mu79,a80,mu80,a81,mu81,a82,mu82,a83,mu83,a84,mu84,a85,mu85,a86,mu86,a87,mu87,a88,mu88,a89,mu89,a90,mu90,a91,mu91,a92,mu92,a93,mu93,a94,mu94,a95,mu95,a96,mu96,a97,mu97,a98,mu98,a99,mu99,a100,mu100,kUmax,kUmin,steep,cliffstart,kR,nves,nsites=10**np.array(model.getParameters())
#%%
t = amici.runAmiciSimulation(model, solver, None).t
R = amici.runAmiciSimulation(model, solver, None).x[:,0]

for n, val in enumerate([20 for i in range(50)]):
    n+=61
    globals()["mu%d"%n] = val


#%% Here, we shift the peak times as we observed that it fits the data better
if (measurements=='real'):
	t_wait = t_wait - 0.0014
nsites =  number_of_sites
model.setParameters(np.log10(np.array([N_var,t_wait,gV,gP,fac,sigma,L,k_base,t0,a1,mu1,a2,mu2,a3,mu3,a4,mu4,a5,mu5,a6,mu6,a7,mu7,a8,mu8,a9,mu9,a10,mu10,a11,mu11,a12,mu12,a13,mu13,a14,mu14,a15,mu15,a16,mu16,a17,mu17,a18,mu18,a19,mu19,a20,mu20,a21,mu21,a22,mu22,a23,mu23,a24,mu24,a25,mu25,a26,mu26,a27,mu27,a28,mu28,a29,mu29,a30,mu30,a31,mu31,a32,mu32,a33,mu33,a34,mu34,a35,mu35,a36,mu36,a37,mu37,a38,mu38,a39,mu39,a40,mu40,a41,mu41,a42,mu42,a43,mu43,a44,mu44,a45,mu45,a46,mu46,a47,mu47,a48,mu48,a49,mu49,a50,mu50,a51,mu51,a52,mu52,a53,mu53,a54,mu54,a55,mu55,a56,mu56,a57,mu57,a58,mu58,a59,mu59,a60,mu60,a61,mu61,a62,mu62,a63,mu63,a64,mu64,a65,mu65,a66,mu66,a67,mu67,a68,mu68,a69,mu69,a70,mu70,a71,mu71,a72,mu72,a73,mu73,a74,mu74,a75,mu75,a76,mu76,a77,mu77,a78,mu78,a79,mu79,a80,mu80,a81,mu81,a82,mu82,a83,mu83,a84,mu84,a85,mu85,a86,mu86,a87,mu87,a88,mu88,a89,mu89,a90,mu90,a91,mu91,a92,mu92,a93,mu93,a94,mu94,a95,mu95,a96,mu96,a97,mu97,a98,mu98,a99,mu99,a100,mu100,kUmax,kUmin,steep,cliffstart,kR,nves,nsites])))

#%%
nom = model.getParameters()

#%%
#define the observable NF_prime
def NF_prime(p):
	model.setParameters(p)
	t = amici.runAmiciSimulation(model, solver, None).t
	R = amici.runAmiciSimulation(model, solver, None).x[:,0]
	(N_var,t_wait,gV,gP,fac,sigma,L,k_base,t0,a1,mu1,a2,mu2,a3,mu3,a4,mu4,a5,mu5,a6,mu6,a7,mu7,a8,mu8,a9,mu9,a10,mu10,a11,mu11,a12,mu12,a13,mu13,a14,mu14,a15,mu15,a16,mu16,a17,mu17,a18,mu18,a19,
	mu19,a20,mu20,a21,mu21,a22,mu22,a23,mu23,a24,mu24,a25,mu25,a26,mu26,a27,mu27,a28,mu28,a29,mu29,a30,mu30,a31,mu31,a32,mu32,a33,mu33,a34,mu34,a35,mu35,a36,mu36,a37,mu37,a38,mu38,a39,mu39,
	a40,mu40,a41,mu41,a42,mu42,a43,mu43,a44,mu44,a45,mu45,a46,mu46,a47,mu47,a48,mu48,a49,mu49,a50,mu50,a51,mu51,a52,mu52,a53,mu53,a54,mu54,a55,mu55,a56,mu56,a57,mu57,a58,mu58,
	a59,mu59,a60,mu60,a61,mu61,a62,mu62,a63,mu63,a64,mu64,a65,mu65,a66,mu66,a67,mu67,a68,mu68,a69,mu69,a70,mu70,a71,mu71,a72,mu72,a73,mu73,a74,mu74,a75,mu75,a76,mu76,a77,mu77,a78,
	mu78,a79,mu79,a80,mu80,a81,mu81,a82,mu82,a83,mu83,a84,mu84,a85,mu85,a86,mu86,a87,mu87,a88,mu88,a89,mu89,a90,mu90,a91,mu91,a92,mu92,a93,mu93,a94,mu94,a95,mu95,a96,mu96,a97,mu97,a98,
	mu98,a99,mu99,a100,mu100,kUmax,kUmin,steep,cliffstart,kR,nves,nsites) = 10**p
	return N_var*(L/(1+exp(-k_base*((t*fac-t_wait)-t0)))+
	a1*exp(-0.5*((t*fac-t_wait)-mu1)**2/sigma**2)+a2*exp(-0.5*((t*fac-t_wait)-mu2)**2/sigma**2)+
	a3*exp(-0.5*((t*fac-t_wait)-mu3)**2/sigma**2)+a4*exp(-0.5*((t*fac-t_wait)-mu4)**2/sigma**2)+
	a5*exp(-0.5*((t*fac-t_wait)-mu5)**2/sigma**2)+a6*exp(-0.5*((t*fac-t_wait)-mu6)**2/sigma**2)+
	a7*exp(-0.5*((t*fac-t_wait)-mu7)**2/sigma**2)+a8*exp(-0.5*((t*fac-t_wait)-mu8)**2/sigma**2)+
	a9*exp(-0.5*((t*fac-t_wait)-mu9)**2/sigma**2)+a10*exp(-0.5*((t*fac-t_wait)-mu10)**2/sigma**2)+
	a11*exp(-0.5*((t*fac-t_wait)-mu11)**2/sigma**2)+a12*exp(-0.5*((t*fac-t_wait)-mu12)**2/sigma**2)+
	a13*exp(-0.5*((t*fac-t_wait)-mu13)**2/sigma**2)+a14*exp(-0.5*((t*fac-t_wait)-mu14)**2/sigma**2)+
	a15*exp(-0.5*((t*fac-t_wait)-mu15)**2/sigma**2)+a16*exp(-0.5*((t*fac-t_wait)-mu16)**2/sigma**2)+a17*exp(-0.5*((t*fac-t_wait)-mu17)**2/sigma**2)+a18*exp(-0.5*((t*fac-t_wait)-mu18)**2/sigma**2)+a19*exp(-0.5*((t*fac-t_wait)-mu19)**2/sigma**2)+a20*exp(-0.5*((t*fac-t_wait)-mu20)**2/sigma**2)+a21*exp(-0.5*((t*fac-t_wait)-mu21)**2/sigma**2)+a22*exp(-0.5*((t*fac-t_wait)-mu22)**2/sigma**2)+a23*exp(-0.5*((t*fac-t_wait)-mu23)**2/sigma**2)+a24*exp(-0.5*((t*fac-t_wait)-mu24)**2/sigma**2)+a25*exp(-0.5*((t*fac-t_wait)-mu25)**2/sigma**2)+a26*exp(-0.5*((t*fac-t_wait)-mu26)**2/sigma**2)+a27*exp(-0.5*((t*fac-t_wait)-mu27)**2/sigma**2)+a28*exp(-0.5*((t*fac-t_wait)-mu28)**2/sigma**2)+a29*exp(-0.5*((t*fac-t_wait)-mu29)**2/sigma**2)+a30*exp(-0.5*((t*fac-t_wait)-mu30)**2/sigma**2)+a31*exp(-0.5*((t*fac-t_wait)-mu31)**2/sigma**2)+a32*exp(-0.5*((t*fac-t_wait)-mu32)**2/sigma**2)+a33*exp(-0.5*((t*fac-t_wait)-mu33)**2/sigma**2)+a34*exp(-0.5*((t*fac-t_wait)-mu34)**2/sigma**2)+a35*exp(-0.5*((t*fac-t_wait)-mu35)**2/sigma**2)+a36*exp(-0.5*((t*fac-t_wait)-mu36)**2/sigma**2)+a37*exp(-0.5*((t*fac-t_wait)-mu37)**2/sigma**2)+a38*exp(-0.5*((t*fac-t_wait)-mu38)**2/sigma**2)+a39*exp(-0.5*((t*fac-t_wait)-mu39)**2/sigma**2)+a40*exp(-0.5*((t*fac-t_wait)-mu40)**2/sigma**2)+a41*exp(-0.5*((t*fac-t_wait)-mu41)**2/sigma**2)+a42*exp(-0.5*((t*fac-t_wait)-mu42)**2/sigma**2)+a43*exp(-0.5*((t*fac-t_wait)-mu43)**2/sigma**2)+a44*exp(-0.5*((t*fac-t_wait)-mu44)**2/sigma**2)+a45*exp(-0.5*((t*fac-t_wait)-mu45)**2/sigma**2)+a46*exp(-0.5*((t*fac-t_wait)-mu46)**2/sigma**2)+a47*exp(-0.5*((t*fac-t_wait)-mu47)**2/sigma**2)+a48*exp(-0.5*((t*fac-t_wait)-mu48)**2/sigma**2)+a49*exp(-0.5*((t*fac-t_wait)-mu49)**2/sigma**2)+a50*exp(-0.5*((t*fac-t_wait)-mu50)**2/sigma**2)+a51*exp(-0.5*((t*fac-t_wait)-mu51)**2/sigma**2)+a52*exp(-0.5*((t*fac-t_wait)-mu52)**2/sigma**2)+a53*exp(-0.5*((t*fac-t_wait)-mu53)**2/sigma**2)+a54*exp(-0.5*((t*fac-t_wait)-mu54)**2/sigma**2)+a55*exp(-0.5*((t*fac-t_wait)-mu55)**2/sigma**2)+a56*exp(-0.5*((t*fac-t_wait)-mu56)**2/sigma**2)+a57*exp(-0.5*((t*fac-t_wait)-mu57)**2/sigma**2)+a58*exp(-0.5*((t*fac-t_wait)-mu58)**2/sigma**2)+a59*exp(-0.5*((t*fac-t_wait)-mu59)**2/sigma**2)+a60*exp(-0.5*((t*fac-t_wait)-mu60)**2/sigma**2)+a61*exp(-0.5*((t*fac-t_wait)-mu61)**2/sigma**2)+a62*exp(-0.5*((t*fac-t_wait)-mu62)**2/sigma**2)+a63*exp(-0.5*((t*fac-t_wait)-mu63)**2/sigma**2)+a64*exp(-0.5*((t*fac-t_wait)-mu64)**2/sigma**2)+a65*exp(-0.5*((t*fac-t_wait)-mu65)**2/sigma**2)+a66*exp(-0.5*((t*fac-t_wait)-mu66)**2/sigma**2)+a67*exp(-0.5*((t*fac-t_wait)-mu67)**2/sigma**2)+a68*exp(-0.5*((t*fac-t_wait)-mu68)**2/sigma**2)+a69*exp(-0.5*((t*fac-t_wait)-mu69)**2/sigma**2)+a70*exp(-0.5*((t*fac-t_wait)-mu70)**2/sigma**2)+a71*exp(-0.5*((t*fac-t_wait)-mu71)**2/sigma**2)+a72*exp(-0.5*((t*fac-t_wait)-mu72)**2/sigma**2)+a73*exp(-0.5*((t*fac-t_wait)-mu73)**2/sigma**2)+a74*exp(-0.5*((t*fac-t_wait)-mu74)**2/sigma**2)+a75*exp(-0.5*((t*fac-t_wait)-mu75)**2/sigma**2)+a76*exp(-0.5*((t*fac-t_wait)-mu76)**2/sigma**2)+a77*exp(-0.5*((t*fac-t_wait)-mu77)**2/sigma**2)+a78*exp(-0.5*((t*fac-t_wait)-mu78)**2/sigma**2)+a79*exp(-0.5*((t*fac-t_wait)-mu79)**2/sigma**2)+a80*exp(-0.5*((t*fac-t_wait)-mu80)**2/sigma**2)+a81*exp(-0.5*((t*fac-t_wait)-mu81)**2/sigma**2)+a82*exp(-0.5*((t*fac-t_wait)-mu82)**2/sigma**2)+a83*exp(-0.5*((t*fac-t_wait)-mu83)**2/sigma**2)+a84*exp(-0.5*((t*fac-t_wait)-mu84)**2/sigma**2)+a85*exp(-0.5*((t*fac-t_wait)-mu85)**2/sigma**2)+a86*exp(-0.5*((t*fac-t_wait)-mu86)**2/sigma**2)+a87*exp(-0.5*((t*fac-t_wait)-mu87)**2/sigma**2)+a88*exp(-0.5*((t*fac-t_wait)-mu88)**2/sigma**2)+a89*exp(-0.5*((t*fac-t_wait)-mu89)**2/sigma**2)+a90*exp(-0.5*((t*fac-t_wait)-mu90)**2/sigma**2)+a91*exp(-0.5*((t*fac-t_wait)-mu91)**2/sigma**2)+a92*exp(-0.5*((t*fac-t_wait)-mu92)**2/sigma**2)+a93*exp(-0.5*((t*fac-t_wait)-mu93)**2/sigma**2)+a94*exp(-0.5*((t*fac-t_wait)-mu94)**2/sigma**2)+a95*exp(-0.5*((t*fac-t_wait)-mu95)**2/sigma**2)+a96*exp(-0.5*((t*fac-t_wait)-mu96)**2/sigma**2)+a97*exp(-0.5*((t*fac-t_wait)-mu97)**2/sigma**2)+a98*exp(-0.5*((t*fac-t_wait)-mu98)**2/sigma**2)+a99*exp(-0.5*((t*fac-t_wait)-mu99)**2/sigma**2)+a100*exp(-0.5*((t*fac-t_wait)-mu100)**2/sigma**2))*R


#%% Definition of the impulse response function
def mEPSC_fun(tstep):
    ###Parameters, don't change!
    size_of_mini = 0.6e-9 #A, Amplitude of mEJC, Estimated from variance-mean of data (see Fig 2F)
    A = -7.209251536449789e-06
    B = 2.709256850482493e-09
    t_0 = 0
    tau_rf = 10.692783377261414
    tau_df =0.001500129264510
    tau_ds = 0.002823055510748#*0.6
    length_of_mini =34*1e-3
    
    """Return one mEPSC."""
    t = np.arange(0,length_of_mini,tstep)
    mEPSC = (t >= t_0)*(A*(1-np.exp(-(t-t_0)/tau_rf))*(B*np.exp(-(t-t_0)/tau_df) + (1-B)*np.exp(-(t-t_0)/tau_ds)))
    mEPSC = -(mEPSC/min(mEPSC) *size_of_mini)
    
    # plt.figure()
    # plt.plot(t,mEPSC)
    # plt.show()
    return mEPSC

#%% We define the current, a convolution of NF_prime and the impulse response function, mepsc
def current(NF_prime):
    timestep = 0.0003912137379126375
    return signal.convolve(NF_prime*timestep, mEPSC_fun(timestep))
t = amici.runAmiciSimulation(model, solver, None).t


#%% Importing and preparation of data
current_data_str = 'data/Current_data_animal'+str(animal)+'.npy'
nf_data_str = 'data/NF_prime_data_animal'+str(animal)+'.npy'

stop_ind=[-9,-4,-5,-7,-9]    #I included here the indices at which to stop counting the peaks in current_data as find_peaks included very tiny peaks
 
if measurements == 'art':
    NF_prime_data = np.load('NF_prime_art.npy')
    current_data=np.load('current_art.npy')
    h=0
    top_peaks_data= find_peaks(current_data)[0]
    bottom_peaks_data = find_peaks(-current_data)[0][:-1]
    NF_prime_peaks_data = find_peaks(NF_prime_data,height=h)[0]
elif measurements == 'real':
    NF_prime_data = np.load(nf_data_str)
    current_data=np.load(current_data_str)
    h=20000
    top_peaks_data= find_peaks(current_data,distance = 10,height=(-3.9e-8,-0.4e-8))[0][:stop_ind[animal-1]]
    bottom_peaks_data = find_peaks(-current_data,3e-8)[0]
    NF_prime_peaks_data = find_peaks(NF_prime_data,height=h)[0]

#%% Definitions of objective functions
from frechetdist import frdist
def frechet_objective(time,measurements,model,t0,tn):
	measurements = measurements.reshape(measurements.size)
	model = model.reshape(model.size)
	time = time.reshape(time.size)
	non_nan_indices  = np.argwhere(~np.isnan(model))
	measurements = measurements[non_nan_indices][t0:tn]
	time = time[non_nan_indices][t0:tn]
	model = model[non_nan_indices][t0:tn]
	model_curve = np.hstack((time,model))
	measurement_curve = np.hstack((time,measurements))
	return(frdist(measurement_curve,model_curve),t0,tn)

if (objective == 'peak_values_NF_prime'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
       	R = amici.runAmiciSimulation(model, solver).x[:,0]
       	t = amici.runAmiciSimulation(model, solver).t
       	NF_prime_model = NF_prime(all_params)
        NF_prime_peaks_model = find_peaks(NF_prime_model)[0]


        return (((NF_prime_data[NF_prime_peaks_data] - 
                         NF_prime_model[NF_prime_peaks_model][:60])**2).sum()) 
                              
if (objective == 'peak_values_current'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
       	R = amici.runAmiciSimulation(model, solver).x[:,0]
       	t = amici.runAmiciSimulation(model, solver).t
       	NF_prime_model = NF_prime(all_params)
        NF_prime_peaks_model = find_peaks(NF_prime_model)[0]
        current_model = current(NF_prime_model)

        top_peaks_model = find_peaks(current_model)[0]
        
        bottom_peaks_model = find_peaks(-current_model)[0]
        return (((current_data[top_peaks_data] - current_model[top_peaks_model][:59])**2).sum()
                + ((current_data[bottom_peaks_data] - current_model[bottom_peaks_model][:60])**2).sum())
                              
                              
if (objective == 'peak_values_NF_prime_and_current'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
        R = amici.runAmiciSimulation(model, solver).x[:,0]
        t = amici.runAmiciSimulation(model, solver).t
        
        NF_prime_model = NF_prime(all_params)
        current_model = current(NF_prime_model)

        top_peaks_model = find_peaks(current_model)[0]
        
        bottom_peaks_model = find_peaks(-current_model)[0]

        NF_prime_peaks_model = find_peaks(NF_prime_model)[0]
        

        return (((current_data[top_peaks_data] - current_model[top_peaks_model][:59])**2).sum()
                + ((current_data[bottom_peaks_data] - current_model[bottom_peaks_model][:60])**2).sum()
                + 1e-16*1e-12 * ((NF_prime_data[NF_prime_peaks_data] - NF_prime_model[NF_prime_peaks_model][:60])**2).sum())

if (objective == 'frechet_NF_prime'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)

        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
        R = amici.runAmiciSimulation(model, solver).x[:,0]
        t = amici.runAmiciSimulation(model, solver).t
       	NF_prime_model = NF_prime(all_params)
        NF_prime_peaks_model = find_peaks(NF_prime_model)[0]
        return (frechet_objective(t,NF_prime_model,NF_prime_data,0,250)[0]+ 
                   frechet_objective(t,NF_prime_model,NF_prime_data,250,500)[0]+
                   frechet_objective(t,NF_prime_model,NF_prime_data,500,750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,750,1000)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1000,1250)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1250,1500)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1500,1750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1750,2000)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2000,2250)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2250,2500)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2500,2750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2750,3000)[0])
    
if (objective == 'frechet_current'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)

        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
        R = amici.runAmiciSimulation(model, solver).x[:,0]
        t = amici.runAmiciSimulation(model, solver).t
       	NF_prime_model = NF_prime(all_params)
        current_model = current(NF_prime_model)
        return (frechet_objective(t,current_model,current_data,0,250)[0]+ 
                   frechet_objective(t,current_model,current_data,250,500)[0]+
                   frechet_objective(t,current_model,current_data,500,750)[0]+
                    frechet_objective(t,current_model,current_data,750,1000)[0]+
                    frechet_objective(t,current_model,current_data,1000,1250)[0]+
                    frechet_objective(t,current_model,current_data,1250,1500)[0]+
                    frechet_objective(t,current_model,current_data,1500,1750)[0]+
                    frechet_objective(t,current_model,current_data,1750,2000)[0]+
                    frechet_objective(t,current_model,current_data,2000,2250)[0]+
                    frechet_objective(t,current_model,current_data,2250,2500)[0]+
                    frechet_objective(t,current_model,current_data,2500,2750)[0]+
                    frechet_objective(t,current_model,current_data,2750,3000)[0])
    
if (objective == 'frechet_NF_prime_and_current'):
    def f_obj(param):
        param_Id = list(model.getParameterIds()[i] for i in param_indices)

        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())
        R = amici.runAmiciSimulation(model, solver).x[:,0]
        t = amici.runAmiciSimulation(model, solver).t
       	NF_prime_model = NF_prime(all_params)
        current_model = current(NF_prime_model)
        return (frechet_objective(t,current_model,current_data,0,250)[0]+ 
                   frechet_objective(t,current_model,current_data,250,500)[0]+
                   frechet_objective(t,current_model,current_data,500,750)[0]+
                    frechet_objective(t,current_model,current_data,750,1000)[0]+
                    frechet_objective(t,current_model,current_data,1000,1250)[0]+
                    frechet_objective(t,current_model,current_data,1250,1500)[0]+
                    frechet_objective(t,current_model,current_data,1500,1750)[0]+
                    frechet_objective(t,current_model,current_data,1750,2000)[0]+
                    frechet_objective(t,current_model,current_data,2000,2250)[0]+
                    frechet_objective(t,current_model,current_data,2250,2500)[0]+
                    frechet_objective(t,current_model,current_data,2500,2750)[0]+
                    frechet_objective(t,current_model,current_data,2750,3000)[0]+
                    frechet_objective(t,current_model,current_data,0,250)[0]+ 
                    1e-12*(frechet_objective(t,NF_prime_model,NF_prime_data,250,500)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,500,750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,750,1000)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1000,1250)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1250,1500)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1500,1750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,1750,2000)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2000,2250)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2250,2500)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2500,2750)[0]+
                    frechet_objective(t,NF_prime_model,NF_prime_data,2750,3000)[0]))

#%% Definition of optimization function
def optimize_params(param_0):
	opt_ = minimize(f_obj, param_0, bounds = param_bounds, method = opt_algotrithm, tol=1e-30).x
	return opt_

#%% Random initialization of parameters 
params_0 = np.vstack([np.log10(np.random.uniform(nom_val/4,4*nom_val,
        n_starts)) for nom_val in 10**(np.array(nom)[param_indices])]).T
print(params_0[0])
np.save(f'{run_dir}/initial_params_array.npy',params_0)


#%% Parallel optimization for all initial values.
if __name__ == "__main__":

    start_time = time.perf_counter()
    with Pool(40) as pool:
        result = pool.map(optimize_params, params_0)
    finish_time = time.perf_counter()
    print("Program finished in {} seconds - using multiprocessing".format(finish_time-start_time))
    print("---")
   
optimized_params = np.array(result)
np.save(f'{run_dir}/optimized_params_array.npy',optimized_params)

#%% Waterfall plot
plt.figure(dpi=500)
obj_values = [f_obj(optimized_params[i]) for i in range(params_0.shape[0])]
sorted_args = np.argsort(obj_values)
sorted_obj = np.sort(obj_values)
plt.plot(range(1,params_0.shape[0]+1),sorted_obj,'.')
plt.xticks(range(1,params_0.shape[0]+1,5))
plt.xlabel('Sorted Run Index')
plt.ylabel('Objective  Value')
plt.savefig(f'{run_dir}/waterfall_plot.png')

#%% Fit plots
min_ind = obj_values.index(min(obj_values))
param_Id = list(model.getParameterIds()[i] for i in param_indices)
model.setParameterById(dict(zip(param_Id, optimized_params[min_ind])))
all_params = model.getParameters()
zoom_name = ['full','zoomed_a','zoomed_b','zoomed_c','zoomed_d','zoomed_e','zoomed_f']
for fit in ['$N\dot{F}$','Current']:
    j=0
    for (n1,n) in [(0,3000),(0,500),(500,1000),(1000,1500),(1500,2000),(2000,2500),(2500,3000)]:
        #data = NF_prime_obs
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, optimized_params[sorted_args[0]])))
        all_params = model.getParameters()
        plt.figure(dpi=350)
        if fit=='Current':
            plt.plot(times[n1:n],current(NF_prime(np.array(all_params)))[n1:n],'-',markersize = 2.5,linewidth = 1.5,label='model')
            plt.plot(times[n1:n],current_data[n1:n],linewidth=0.9,label='data')
        elif fit=='$N\dot{F}$':
            plt.plot(times[n1:n],NF_prime(np.array(all_params))[n1:n],'-',markersize = 2.5,linewidth = 1.5,label='model')
            plt.plot(times[n1:n],NF_prime_data[n1:n],linewidth=0.9,label='data')           
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        #model.setParameterById(dict(zip(param_Id, params_0[i])))
        #all_params = model.getParameters()
        #plt.plot(times[:],NF_prime(np.array(all_params))[:],'.',markersize = 0.5,linewidth = 1.1)
        plt.title('Model fit')
        plt.xlabel('Time ($t$)')
        plt.ylabel(f'{fit}')
        plt.legend()
        plt.savefig(f'{run_dir}/best_fit_plot_{fit}_{zoom_name[j]}.png')
        j+=1

#%% Saving the best NF_prime and current values
np.save(f'{run_dir}/NF_prime_best.npy',NF_prime(np.array(all_params)))
np.save(f'{run_dir}/Current_best.npy',current(NF_prime(np.array(all_params))))

np.save(f'{run_dir}/optimized_objective__array.npy',np.array(obj_values))
with open(f'{run_dir}/min_ind.txt', 'w') as f:
    f.write('min_ind = ' + str(min_ind) + '\n' + 'min_obj=' + str(sorted_obj[0])+ '\n') 

#%% Plot parrameter values    
colors = ['tab:green', 'tab:blue','tab:orange','tab:brown']+['gainsboro']*50
plt.figure(dpi=500)
plt.rc('axes', prop_cycle=(cycler('color', colors)))


plt.plot(optimized_params[sorted_args[-5:-1]].T,list(model.getParameterIds()[i] for i in param_indices),
         '.--',color = 'gainsboro',markersize=9)
plt.plot(optimized_params[sorted_args[-1]].T,list(model.getParameterIds()[i] for i in param_indices),
         '.--',markersize=9,color = 'gainsboro', label='worst_fit')
plt.plot(optimized_params[sorted_args[1:5]].T,list(model.getParameterIds()[i] for i in param_indices),
         '.--',markersize=12)
plt.plot(optimized_params[sorted_args[0]].T,list(model.getParameterIds()[i] for i in param_indices),
         '*-',markersize=10,color = 'tab:red', label='best_fit')
plt.ylabel('Parameter')
plt.xlabel('Parameter Value')
plt.title('Estimated Parameters')
plt.legend()
plt.savefig(f'{run_dir}/estimated_parameters.png')

if measurements == 'art':
    plt.figure(dpi=500)
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    
    
    plt.plot(optimized_params[sorted_args[-5:-1]].T,list(model.getParameterIds()[i] for i in param_indices),
             '.--',color = 'gainsboro',markersize=9)
    plt.plot(optimized_params[sorted_args[-1]].T,list(model.getParameterIds()[i] for i in param_indices),
             '.--',markersize=9,color = 'gainsboro', label='worst_fit')
    plt.plot(optimized_params[sorted_args[1:5]].T,list(model.getParameterIds()[i] for i in param_indices),
             '.--',markersize=12)
    plt.plot(optimized_params[sorted_args[0]].T,list(model.getParameterIds()[i] for i in param_indices),
             '*-',markersize=10,color = 'tab:red', label='best_fit')
    plt.plot(np.array(nom)[param_indices].T,list(model.getParameterIds()[i] for i in param_indices),
                 '+-',color='black', markersize=15,label='actual_values')
        
    plt.ylabel('Parameter')
    plt.xlabel('Parameter Value')
    plt.title('Estimated Parameters')
    plt.legend()
    plt.savefig(f'{run_dir}/estimated_parameters_with_actual_values.png')

#%% Write values of optimized pparameters to file
params = list(model.getParameterIds()[i] for i in param_indices)
with open(f'{run_dir}/best_parameters.txt', 'w') as f:
    for i in range(len(params)):
        f.write(params[i] + ' = ' + str(10**optimized_params[sorted_args[0]][i]) + '\n') 
