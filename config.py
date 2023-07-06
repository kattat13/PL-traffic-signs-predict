class Config(object):

    SECRET_KEY = 'kasia'
    SESSION_COOKIE_SECURE = True
    DEFAULT_THEME = None


class DevelopmentConfig(Config):
    DEBUG = True
