import sklearn.preprocessing as sklpre
import sklearn.metrics as sklmet


def multilabel_f1(y_true,y_pred):
    m = sklpre.MultiLabelBinarizer().fit(y_true)

    return sklmet.f1_score(m.transform(y_true),
         m.transform(y_pred),
         average='macro')
