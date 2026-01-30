# CS505: NLP - Spring 2026

def calculate_accuracy(predictions, labels):
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

def macro_f1(predictions, labels, num_classes=4):
    pass
    # TODO: implement the macro-F1 score.
    # Recall that this involves computing the F1 score separately for
    # each label, and then taking the macroaverage. Return the macro-F1
    # score as a floating-point number.
    # STUDENT START --------------------------------------
    f1_scores = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(predictions, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(predictions, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(predictions, labels) if p != c and l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)
    # STUDENT END -------------------------------------------