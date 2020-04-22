# # model specification
model = DNN([128, 28, 10], ['esp', 'esp', 'softmax'])

# # train
model, betaL = train(x_train, y_train, model, obj, 'crossEntropy', 3000, 0.1, [4], 100)

# # evaluation
cost, accuracy = obj(model(x_train), y_train, 'crossEntropy'), eval_accuracy(model(x_train), y_train, 'logistic')

# plot code

plotNode(betaL, 4, 1)