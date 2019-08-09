import configparser


class ConfigReader:
    config = configparser.ConfigParser()

    @staticmethod
    def readconfig(section, key):
        ConfigReader.config.read('Config.ini')
        return ConfigReader.config[section][key]
