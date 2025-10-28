import os

from cag.utils import config
from dotenv import load_dotenv

from graph.graph_creators.gc_geoservice import GeoserviceGraphCreator
from graph.graph_creators.gc_openalex import OpenAlexGraphCreator
from graph.graph_creators.gc_pangaea import PangaeaGraphCreator
from graph.timer import Timer

load_dotenv()

config_ = config.Config(
    # url=os.getenv("CAG_ARANGO_HOST") or "127.0.0.1:8529",
    # user="root",
    password=os.getenv("CAG_ARANGO_ROOT_PASSWORD") or "root",
    database="ScienceSearch",
    graph="ows-eo-kg",
)

t = Timer()
t.start()

GeoserviceGraphCreator(
    corpus_file_or_dir="/localdata1/proj_ows/eoc_geoservice/",
    conf=config_,
    initialize=True,
    load_generic_graph=False,
)

PangaeaGraphCreator(
    corpus_file_or_dir="pangaea/",
    conf=config_,
    initialize=True,
    load_generic_graph=False,
)

OpenAlexGraphCreator(
    corpus_file_or_dir="/localdata1/proj_ows/openalex/tagged_and_filtered_snapshot/",
    conf=config_,
    initialize=True,
    load_generic_graph=False,
)

t.stop()
