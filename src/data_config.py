class DataConfigs:
    Android = {
        'name': 'Android',
        'structured_path': '../../data/structured/Android_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    }
    Apache = {
        'name': 'Apache',
        'structured_path': '../../data/structured/Apache_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    BGL = {
        'name': 'BGL',
        'structured_path': '../../data/structured/BGL_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
    }
    Hadoop = {
        'name': 'Hadoop',
        'structured_path': '../../data/structured/Hadoop_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    HDFS = {
        'name': 'HDFS',
        'structured_path': '../../data/structured/HDFS_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    }
    HealthApp = {
        'name': 'HealthApp',
        'structured_path': '../../data/structured/HealthApp_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
    }
    HPC = {
        'name': 'HPC',
        'structured_path': '../../data/structured/HPC_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
    }
    Linux = {
        'name': 'Linux',
        'structured_path': '../../data/structured/Linux_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    }
    Mac = {
        'name': 'Mac',
        'structured_path': '../../data/structured/Mac_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenSSH = {
        'name': 'OpenSSH',
        'structured_path': '../../data/structured/OpenSSH_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenStack = {
        'name': 'OpenStack',
        'structured_path': '../../data/structured/OpenStack_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
    }
    Proxifier = {
        'name': 'Proxifier',
        'structured_path': '../../data/structured/Proxifier_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    }
    Spark = {
        'name': 'Spark',
        'structured_path': '../../data/structured/Spark_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    }
    Thunderbird = {
        'name': 'Thunderbird',
        'structured_path': '../../data/structured/Thunderbird_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    Windows = {
        'name': 'Windows',
        'structured_path': '../../data/structured/Windows_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
    }
    Zookeeper = {
        'name': 'Zookeeper',
        'structured_path': '../../data/structured/Zookeeper_2k.log_structured.csv',
        'unstructured_path': '../../data/unstructured/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    }
