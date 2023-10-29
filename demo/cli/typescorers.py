import sklearn.metrics

def type_1_error_ratio(y_true, y_pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan

def type_2_error_ratio(y_true, y_pred):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan

type_1_error_scorer = sklearn.metrics.make_scorer(type_1_error_ratio, greater_is_better=False)
type_2_error_scorer = sklearn.metrics.make_scorer(type_2_error_ratio, greater_is_better=False)

