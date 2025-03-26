import pandas as pd
from typing import List, Dict

class AgentConfig:
    def __init__(self):
        self.data_sources = {
            'faq': 'data/faq.csv',
            'products': 'data/product_info.csv',
            'support_docs': 'data/support_docs.txt'
        }
    
    def load_data_sources(self) -> Dict[str, pd.DataFrame]:
        loaded_sources = {}
        for name, path in self.data_sources.items():
            if path.endswith('.csv'):
                loaded_sources[name] = pd.read_csv(path)
            elif path.endswith('.txt'):
                with open(path, 'r') as f:
                    loaded_sources[name] = f.read()
        return loaded_sources