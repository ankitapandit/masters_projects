import random

random = random.SystemRandom()


def get_random_string(length=12, allowed_chars='abcdefghijklmnopqrstuvwxyz''ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    return ''.join(random.choice(allowed_chars) for i in range(length))


def get_secret_key():
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return get_random_string(32, chars)


def main():
    secret_key = get_secret_key()
    print(secret_key)


if __name__ == "__main__":
    main()
