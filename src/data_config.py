class DataConfigs:
    Android = {
        'name': 'Android',
        'assignments_path': 'data/samples_2k/structured/Android_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    }
    Apache = {
        'name': 'Apache',
        'assignments_path': 'data/samples_2k/structured/Apache_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    BGL = {
        'name': 'BGL',
        'assignments_path': 'data/samples_2k/structured/BGL_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
    }
    Hadoop = {
        'name': 'Hadoop',
        'assignments_path': 'data/samples_2k/structured/Hadoop_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    HDFS = {
        'name': 'HDFS',
        'assignments_path': 'data/samples_2k/structured/HDFS_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    }
    HealthApp = {
        'name': 'HealthApp',
        'assignments_path': 'data/samples_2k/structured/HealthApp_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
    }
    HPC = {
        'name': 'HPC',
        'assignments_path': 'data/samples_2k/structured/HPC_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
    }
    Linux = {
        'name': 'Linux',
        'assignments_path': 'data/samples_2k/structured/Linux_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    }
    Mac = {
        'name': 'Mac',
        'assignments_path': 'data/samples_2k/structured/Mac_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenSSH = {
        'name': 'OpenSSH',
        'assignments_path': 'data/samples_2k/structured/OpenSSH_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    }
    OpenStack = {
        'name': 'OpenStack',
        'assignments_path': 'data/samples_2k/structured/OpenStack_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
    }
    Proxifier = {
        'name': 'Proxifier',
        'assignments_path': 'data/samples_2k/structured/Proxifier_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    }
    Spark = {
        'name': 'Spark',
        'assignments_path': 'data/samples_2k/structured/Spark_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    }
    Thunderbird = {
        'name': 'Thunderbird',
        'assignments_path': 'data/samples_2k/structured/Thunderbird_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
    }
    Windows = {
        'name': 'Windows',
        'assignments_path': 'data/samples_2k/structured/Windows_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
    }
    Zookeeper = {
        'name': 'Zookeeper',
        'assignments_path': 'data/samples_2k/structured/Zookeeper_2k.log_structured.csv',
        'unstructured_path': 'data/samples_2k/unstructured/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    }
    BGL_FULL = {
        'name': 'BGL',
        'unstructured_path': 'data/full/unstructured/BGL_filtered_reduced.log',
        'assignments_path': 'data/full/assignments/BGL_filtered_reduced_assignments.csv',
        'template_path': 'data/full/templates/BGL_templates.csv',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
    }
