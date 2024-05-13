## Predicting Police Misconduct 
![](<assets/UChicago_UCL_CrimeLab_Color RGB.png>)

This repository contains code to replicate analysis found in our research paper [Predicting Police Misconduct](https://www.nber.org/papers/w32432). 

In 2016, the University of Chicago Crime Lab partnered with the Chicago Police Department (CPD) to build an early intervention system based on a statistical analysis of 10+ years of CPD data. Our goal was to create a more accurate EIS by using statistics and machine learning to discover the most predictive risk factors, and to understand the extent to which risk of misconduct could be predicted in the first place. 

Using estimates from a data-driven algorithm that predicts an officer’s future risk of serious misconduct from their past record of activity and complaints against them, we find that the top 2% of officers with highest predicted risk are 6 times more likely to engage in serious misconduct than the average officer. While this level of predictability is far from perfect, it provides an enormously helpful decision aid for targeting supportive resources, enabling the Chicago Police Department to direct supervisor time and training and mental health services to those who will benefit most.

Our work revealed several key lessons relevant both to the Chicago Police Department and other departments around the country. 

- A data-driven system can identify officers at significantly elevated risk for misconduct, but the level of accuracy is far from perfect. While predictive models can help prevent misconduct, no EIS will be a panacea.
- Predicted risk of misconduct is not simply a proxy for policing activity. Officers with very similar levels of measured activity (arrests, guns confiscated, etc.) can vary enormously in their risk of future misconduct.
- Risk of on-duty and off-duty misconduct are correlated. This suggests that officer wellness interventions may help reduce both on-duty and off-duty adverse events.  
- While EI systems often focus on ‘serious events’ as warning signs, our data analysis suggests what matters more is an officer’s larger pattern of events. For example, an officer’s entire record of complaints is significantly more predictive than just their record of prior sustained complaints. 
Police departments can get most of the benefits of a fully-blown predictive model at much lower cost with a simple policy based on count of prior complaints from the past two years. 

To learn more about this work, please visit the project's website https://crimelab.uchicago.edu/projects/officer-support-system-oss/

## Guide to this repository

This repo contains two separate analyses. 

The `nypd_replication` folder contains all materials, including data and code, to replicate the supplemental analysis conducted with public NYPD data. This analysis replicates are two main findings that misconduct is predictable (with roughly the same level of accuracy) and that the accuracy of the simple 'rank by complaints' policy compares favorably to more complex machine learning models. 

The `police_violence_and_agency_size` folder contains all materials (code and data) to replicate our brief analysis that police departments with fewer than 500 officers account for the majority (~62%) of police killings, suggesting that the challenges of police misconduct are not limited to large police departments. 


