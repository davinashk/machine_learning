import tiktoken

encoder = tiktoken.get_encoding("gpt")
encoded = encoder.encode("Hello, world!")
print(encoded)
print(encoder.decode(encoded))

