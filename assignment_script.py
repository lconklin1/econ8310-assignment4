import pandas as pd
import arviz as az
import pymc as pm

def run_model():
    df = pd.read_csv('https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv')

    gate_30_data = df[df['version'] == 'gate_30']
    gate_30_retain_success = gate_30_data['retention_1'].sum()
    gate_30_retain_failures = gate_30_data.shape[0] - gate_30_retain_success

    gate_40_data = df[df['version'] == 'gate_40']
    gate_40_retain_success = gate_40_data['retention_1'].sum()
    gate_40_retain_failures = gate_40_data.shape[0] - gate_40_retain_success

    with pm.Model() as model:
        g_30_retain = pm.Beta('g_30_retain', alpha=1, beta=1)
        g_40_retain = pm.Beta('g_40_retain', alpha=1, beta=1)

        gate_30_retained = pm.Binomial('gate_30_retained',
            n=gate_30_retain_success + gate_30_retain_failures, 
            p = g_30_retain, 
            observed = gate_30_retain_success)
        
        gate_40_retained = pm.Binomial('gate_40_retained',
            n=gate_40_retain_success + gate_40_retain_failures, 
            p = g_40_retain, 
            observed = gate_40_retain_success)
                                    
        delta = pm.Deterministic('delta',g_40_retain-g_30_retain)
        trace_retention1 = pm.sample(2000,return_inferencedata=True)
    az.summary(trace_retention1, hdi_prob=0.95)

    az.plot_posterior(trace_retention1, var_names = ['delta'])

    deltas = trace_retention1.posterior['delta'].values.flatten()
    prob_delta_greater_than_zero = (deltas > 0).mean()
    prob_delta_less_than_zero = (deltas < 0).mean()

    print('Probability that moving from gate level 30 to gate level 40 increased 1-day retention: ' + str(prob_delta_greater_than_zero*100) + '%')
    print('Probability that moving from gate level 30 to gate level 40 decreased 1-day retention: ' + str(prob_delta_less_than_zero*100) + '%')


    gate_30_retain_success_7day = gate_30_data['retention_7'].sum()
    gate_30_retain_failure_7day = gate_30_data.shape[0] - gate_30_retain_success

    gate_40_retain_success_7day = gate_40_data['retention_7'].sum()
    gate_40_retain_failure_7day = gate_40_data.shape[0] - gate_40_retain_success

    with pm.Model() as model:
        g_30_retain_7day = pm.Beta('g_30_retain_7day', alpha=1, beta=1)
        g_40_retain_7day = pm.Beta('g_40_retain_7day', alpha=1, beta=1)

        customer_retained_7 = pm.Binomial('customer_retained_7', n=gate_30_retain_success_7day + gate_30_retain_failure_7day, 
                                p = g_30_retain_7day, 
                                observed = gate_30_retain_success_7day)
        customer_not_retained_7 = pm.Binomial('customer_not_retained_7', n=gate_30_retain_success_7day + gate_30_retain_failure_7day, 
                                p = g_40_retain_7day, 
                                observed = gate_30_retain_success_7day)
                                    
        delta = pm.Deterministic('delta',g_40_retain_7day - g_30_retain_7day)
        trace_retention7 = pm.sample(2000,return_inferencedata=True)
    az.summary(trace_retention7, hdi_prob=0.95)
    az.plot_posterior(trace_retention7, var_names = ['delta'])

    deltas_7day = trace_retention7.posterior['delta'].values.flatten()
    prob_delta_greater_than_zero_7day = (deltas_7day > 0).mean()
    prob_delta_less_than_zero_7day = (deltas_7day < 0).mean()

    print('Probability that moving from gate level 30 to gate level 40 increased 7-day retention: ' + str(prob_delta_greater_than_zero_7day*100) + '%')
    print('Probability that moving from gate level 30 to gate level 40 decreased 7-day retention: ' + str(prob_delta_less_than_zero_7day*100) + '%')

if __name__ =='__main__':
    run_model()
