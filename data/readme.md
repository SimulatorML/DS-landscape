# Data structure

- `hh_parsed_folder` - row parsed data from hh.ru
- `processed/vacancies.csv` - collected vacancies after preprocesing
- `features` - features
	- `skills.txt`: all skills after corrections ordered by name
	- `skill_index_to_corrected.pkl`: dictionary skill index to corrected skill name
	- `skill_original_to_index.pkl`: dictionary skill name to index
	- `skills.csv`: dataframe with columns: [`skill_name`, `skill_id`, `salary_q25`, `salary_q50`, `salary_q75`, `frequency`, `popular_profession_id`, `popular_profession_name`, `<professions>`]
	- `prof.csv`: dataframe with columns: [`prof_name`, `prof_id`, `salary_q25`, `salary_q50`, `salary_q75`, `frequency`, `popular_skills`]
	- `matrix.csv`: skill-profession relationship matrix (numpy)