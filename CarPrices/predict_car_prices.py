from car_prices import predict


if __name__ == "__main__":
    example_inputs = [
        [5, 2000, 1],
        [10, 4000, 1],
    ]
    preds = predict(example_inputs)
    print(preds)

