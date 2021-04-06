"""Functions for dealing with Mallet's diagnostics and state files
"""
import xml.etree.ElementTree as ET
import pandas as pd

# Diagnostics XML constants
TOPIC="topic"
TOPIC_ID="id"


def diagnostics_xml_to_dataframe(xml_path):
    """Returns a pandas dataframe with the
    :param xml_path: Path to a Mallet diagnostics file
    """
    model_root = ET.parse(xml_path).getroot()
    topic_metrics = list()
    for t in model_root.findall(TOPIC):
        attributes = {}
        for k,v in t.items():
            if k == TOPIC_ID:
                attributes[k] = int(v)
            else:
                attributes[k] = float(v)
        topic_metrics.append(attributes)

    return pd.DataFrame.from_records(topic_metrics, index=TOPIC_ID)

