class DataConfigs:
    Android = {
        'name': 'Android',
        'path': '../../data/Android_2k.log_structured.csv',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    }
    Apache = {
        'name': 'Apache',
        'path': '../../data/Apache_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+'],
    }
    BGL = {
        'name': 'BGL',
        'path': '../../data/BGL_2k.log_structured.csv',
        'regex': [r'core\.\d+'],
    }
    Hadoop = {
        'name': 'Hadoop',
        'path': '../../data/Hadoop_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+'],
    }
    HDFS = {
        'name': 'HDFS',
        'path': '../../data/HDFS_2k.log_structured.csv',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    }
    HealthApp = {
        'name': 'HealthApp',
        'path': '../../data/HealthApp_2k.log_structured.csv',
        'regex': [],
    }
    HPC = {
        'name': 'HPC',
        'path': '../../data/HPC_2k.log_structured.csv',
        'regex': [r'=\d+'],
    }
    Linux = {
        'name': 'Linux',
        'path': '../../data/Linux_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    }
    Mac = {
        'name': 'Mac',
        'path': '../../data/Mac_2k.log_structured.csv',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenSSH = {
        'name': 'OpenSSH',
        'path': '../../data/OpenSSH_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenStack = {
        'name': 'OpenStack',
        'path': '../../data/OpenStack_2k.log_structured.csv',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
    }
    Proxifier = {
        'name': 'Proxifier',
        'path': '../../data/Proxifier_2k.log_structured.csv',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    }
    Spark = {
        'name': 'Spark',
        'path': '../../data/Spark_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    }
    Thunderbird = {
        'name': 'Thunderbird',
        'path': '../../data/Thunderbird_2k.log_structured.csv',
        'regex': [r'(\d+\.){3}\d+'],
    }
    Windows = {
        'name': 'Windows',
        'path': '../../data/Windows_2k.log_structured.csv',
        'regex': [r'0x.*?\s'],
    }
    Zookeeper = {
        'name': 'Zookeeper',
        'path': '../../data/Zookeeper_2k.log_structured.csv',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    }
