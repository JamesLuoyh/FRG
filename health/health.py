import numpy as np
import pandas as pd


def create_health_dataset(batch=64):
    d = pd.read_csv('health.csv')
    d = d[d['YEAR_t'] == 'Y3']
    sex = d['sexMISS'] == 0
    age = d['age_MISS'] == 0
    d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    d = d[sex & age]

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    ages = d[['age_%d5' % (i) for i in range(0, 9)]]
    age_05 = (d['age_05'] == 1).copy()
    age_15 = (d['age_15'] == 1).copy()
    age_25 = (d['age_25'] == 1).copy()
    age_35 = (d['age_35'] == 1).copy()
    age_45 = (d['age_45'] == 1).copy()
    age_55 = (d['age_55'] == 1).copy()
    age_65 = (d['age_65'] == 1).copy()
    age_75 = (d['age_75'] == 1).copy()
    age_85 = (d['age_85'] == 1).copy()
    # sexs = d[['sexMALE', 'sexFEMALE']]
    charlson = (d['CharlsonIndexI_max']).copy()
    # print((charlson == 0).sum())
    d = d.drop(
        ['age_%d5' % (i) for i in range(0, 9)] + ['sexMISS', 'age_MISS', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1)
    d=(d-d.min())/(d.max()-d.min())
    d['L'] = (age_05 | age_15 | age_25 | age_35 | age_45).astype(np.float32)
    d['H'] = (age_55 | age_65 | age_75 | age_85).astype(np.float32)
    d['s'] = d['H']
    d['y'] = (charlson > 0).astype(np.float32)
    d['label'] = (charlson > 0).astype(np.float32)
    # print(len(d.columns))
    d = d.fillna(-1)
    print(len(d.columns))
    d.to_csv('health_normalized.csv', index=False, header=False)
    # print(d.columns)
    # x = d.to_numpy()[:-5]
    # print(x.shape)
    # labels = gather_labels(x)
    # print(len(labels))
    # xs = np.zeros_like(x)
    # for i in range(len(labels)):
    #     xs[:, i] = x[:, i] > labels[i]
    # # x = xs[:, np.nonzero(np.mean(xs, axis=0) > 0.05)[0]].astype(np.float32)
    # x = xs[:, np.mean(xs, axis=0) > 0.05].astype(np.float32)

    # print(x.shape)
    # N = pd.DataFrame(x)
    # print('print(d[L].sum())', d['L'].sum())
    # N['L'] = d['L']
    # print('print(N[L].sum())', N['L'].sum())
    # N['H'] = d['H']
    # N['s'] = d['s']
    # N['y'] = d['y']
    # N['label'] = d['label']
    # print(len(N))
    # N.to_csv('health_normalized.csv', index=False)
    # u = np.expand_dims(sexs.as_matrix()[:, 0], 1)
    # v = ages.as_matrix()
    # u = np.concatenate([v, u], axis=1).astype(np.float32)
    # y = (charlson.as_matrix() > 0).astype(np.float32)

    # idx = np.arange(y.shape[0])
    # np.random.shuffle(idx)
    # cf = int(0.8 * d.shape[0])
    # train_ = (x[idx[:cf]], u[idx[:cf]], y[idx[:cf]])
    # test_ = (x[idx[cf:]], u[idx[cf:]], y[idx[cf:]])
    # train = tf.data.Dataset.from_tensor_slices(train_).shuffle(cf).batch(batch)
    # test = tf.data.Dataset.from_tensor_slices(test_).batch(batch)

    # class Distribution():
    #     def __init__(self):
    #         self.pv = tfd.Bernoulli(probs=np.mean(train_[1][:, -1]))
    #         self.pu = tfd.Multinomial(total_count=1., probs=np.mean(train_[1][:, :-1], axis=0))

    #     def log_prob(self, u):
    #         return self.pv.log_prob(u[:, -1]) + self.pu.log_prob(u[:, :-1])

    # dist = Distribution()
    # return train, test, dist

create_health_dataset()
