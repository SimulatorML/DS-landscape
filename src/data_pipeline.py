from src.data.ParserApiHH import ParserApiHH
from src.data.preprocessing import Preprocessor
from src.features.features_processor import FeaturesProcessor

# hh parsing
dc = ParserApiHH()
# this is a FIRST stage
parsed_ids = dc.get_parsed_ids()
dc.process_ids(parsed_ids)
# this is a SECOND stage
df, filename = dc.load_vacancies_ids()
dc.process_vacancies_chunked(df, filename, specify_chunks=None)

# proprocessing
Preprocessor().process()

# features
FeaturesProcessor().process()
