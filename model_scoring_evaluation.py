import numpy as np
import os
import datetime

np.set_printoptions(precision=5)
import utils


def score_all_predictions(pred_file, date, model_date, mse=False, key='cases', bin_cutoffs=[20, 1000]):
    true_data = utils.get_processed_df('nyt_us_counties_daily.csv')
    cum_data = utils.get_processed_df('nyt_us_counties.csv')
    proc_score_date = utils.process_date(date, true_data)
    proc_model_date = utils.process_date(model_date, true_data)
    raw_pred_data = np.genfromtxt(pred_file, delimiter=',', skip_header=1, dtype=np.str)
    date_preds = np.array([row for row in raw_pred_data if date in row[0]])
    all_fips = np.array([row[0].split('-')[-1] for row in date_preds])
    all_preds = date_preds[:, 1:].astype(np.float)
    true_data = np.array([utils.get_region_data(true_data, fips, proc_date=proc_score_date, key=key) for fips in all_fips])
    cum_data = np.array([utils.get_region_data(cum_data, fips, proc_date=proc_model_date, key=key) for fips in all_fips])
    return get_scores(all_fips, all_preds, true_data, cum_data, mse=mse, bin_cutoffs=bin_cutoffs)


def get_scores(all_fips, all_preds, true_data, cum_data, bin_cutoffs=[20, 1000], mse=False):
    tot_loss = 0
    bin_losses, bin_counts = np.zeros(len(bin_cutoffs) + 1), np.zeros(len(bin_cutoffs) + 1)
    for fips, preds, true_number, cum_number in zip(all_fips, all_preds, true_data, cum_data):
        if mse:
            loss = (preds[4] - true_number) ** 2
        else:
            loss = pinball_loss(preds, true_number)
        tot_loss += loss
        done = False
        for i, bc in enumerate(bin_cutoffs):
            if cum_number <= bc:
                bin_losses[i] += loss
                bin_counts[i] += 1
                done = True
                break
        if not done:
            bin_losses[-1] += loss
            bin_counts[-1] += 1

    return tot_loss / len(all_preds), bin_losses / bin_counts


def pinball_loss(preds, true_val, p_vals=np.arange(0.1, 1.0, 0.1)):
    loss = 0
    for pred, p in zip(preds, p_vals):
        delta = np.abs(true_val - pred)
        if pred < true_val:
            loss += p * delta
        else:
            loss += (1 - p) * delta
    return loss / len(p_vals)


def generate_day_tag(start_date, predicted_length):
    first_day = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    labels = []
    for i in range(predicted_length):
        day_to_add = datetime.timedelta(days=i)
        label_name = (day_to_add + first_day).strftime('%Y-%m-%d')
        labels.append(label_name)
    return labels


if __name__ == '__main__':
    pred_file = '../submissions/submission_lr_burning.csv'
    # pred_file = '../sample_submission.csv'

    # dt: pinball = 0.533177, mse = 86.064536
    # pinball = 0.332122, mse = 25.215638

    # for consecutive several days
    start_date = '2020-05-18'
    predicted_length = 7
    date_list = generate_day_tag(start_date, predicted_length)
    for day in date_list:
        scores = score_all_predictions(pred_file, day, '2020-05-17', key='deaths')
        scores_mse = score_all_predictions(pred_file, day, '2020-05-17', key='deaths', mse=True)
        print("Day %s: pinball=%f mse=%f" % (day, scores[0], scores_mse[0]))

    # for single day
    # scores = score_all_predictions(pred_file, '2020-04-28', '2020-04-27', key='deaths')
    # scores_mse = score_all_predictions(pred_file, '2020-04-28', '2020-04-27', key='deaths', mse=True)
    # print("pinball=%f mse=%f" % (scores[0], scores_mse[0]))