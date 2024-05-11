SUBDIRS := clean_complaints_and_allegations \
	       clean_lawsuits \
           clean_roster \
           create_career_start_end_dates \
		   create_observations_main_table \
		   create_features_and_outcomes \
		   train_models
	
all: $(SUBDIRS)
.PHONY: $(SUBDIRS)


clean_complaints_and_allegations:
	cd clean_complaints_and_allegations && python clean_complaints_and_allegations.py

clean_lawsuits:
	cd clean_lawsuits && python clean_lawsuits.py

clean_roster:
	cd clean_roster && python clean_roster.py

create_career_start_end_dates:
	cd create_career_start_end_dates && python create_career_start_end_dates.py

create_observations_main_table:
	cd create_observations_main_table && python create_observation_table.py

create_features_and_outcomes:
	cd create_features_and_outcomes && python create_features_and_outcomes.py

train_models:
	cd train_models && python train_models.py --mc_iters 1 --n_jobs 3