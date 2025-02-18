from .model_1 import evaluate_1
from .model_2 import evaluate_2
from .model_3 import evaluate_3
from .model_4 import evaluate_4
from .model_5 import evaluate_5
from .model_6 import evaluate_6
from .model_7 import evaluate_7
from .model_8 import evaluate_8
from .model_9 import evaluate_9
import torch
import wandb


def get_evaluate(model, device, cfgs, data_path, logger, epoch, val_eval):
    torch.cuda.empty_cache()
    model.eval()
    model.to(device)
    with torch.no_grad():
        if cfgs['evaluator_type'] == 1:
            neighbor_num = 25
            stats = evaluate_1(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 2:
            neighbor_num = 500
            stats = evaluate_2(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 3:
            neighbor_num = 25
            stats = evaluate_3(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 4:
            neighbor_num = 25
            stats = evaluate_4(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 5:
            neighbor_num = 60
            stats = evaluate_5(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 6:
            neighbor_num = 25
            stats = evaluate_6(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 8:
            neighbor_num = 60
            stats = evaluate_8(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 9:
            neighbor_num = 60
            stats = evaluate_9(model, device, cfgs, data_path, val_eval)
        else:
            raise ValueError('Invalid evaluator type: {}'.format(cfgs['evaluator_type']))
    
    logger.info(f'final model of epoch {epoch}:')

    metrics = {}
    metrics['ave_one_percent_recall'] = {}
    metrics['average_sim'] = {}
    metrics['ave_recall'] = {}
    for i in range(neighbor_num):
        metrics['ave_recall'][str(i)] = {}

    for database_name in stats:
        logger.info('Dataset: {}'.format(database_name))
        metrics['ave_one_percent_recall'][database_name] = 0.0
        metrics['average_sim'][database_name] = 0.0
        for i, _ in metrics['ave_recall'].items():
            metrics['ave_recall'][i][database_name] = 0.0
        query_type_num = 0.0
        for query_type in stats[database_name]:
            logger.info('current way of query to database: {}'.format(query_type))
            t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
            logger.info(t.format(stats[database_name][query_type]['ave_one_percent_recall'],
                                    stats[database_name][query_type]['average_sim']))
            logger.info(stats[database_name][query_type]['ave_recall'])
            metrics['ave_one_percent_recall'][database_name] += float(stats[database_name][query_type]['ave_one_percent_recall'])
            metrics['average_sim'][database_name] += float(stats[database_name][query_type]['average_sim'])
            for i, v in enumerate(stats[database_name][query_type]['ave_recall']):
                metrics['ave_recall'][str(i)][database_name] += float(v)
            query_type_num += 1.0
        metrics['ave_one_percent_recall'][database_name] /= max(1.0, query_type_num)
        metrics['average_sim'][database_name] /= max(1.0, query_type_num)
        for i, _ in metrics['ave_recall'].items():
            metrics['ave_recall'][i][database_name] /= max(1.0, query_type_num)

    if cfgs['evaluator_type'] == 5 or cfgs['evaluator_type'] == 8 or cfgs['evaluator_type'] == 9:
        ave_recall_name = 'ave_recall'
        ave_one_percent_recall_name = 'ave_one_percent_recall'
        for database_name in stats:
            for query_type in stats[database_name]:
                logger.info(f'{stats[database_name][query_type][ave_recall_name][0]:.2f}   '
                            f'{stats[database_name][query_type][ave_recall_name][4]:.2f}   '
                            f'{stats[database_name][query_type][ave_one_percent_recall_name]:.2f}   ')
        for database_name in stats:
            for query_type in stats[database_name]:
                print(f'{stats[database_name][query_type][ave_recall_name][0]:.2f}   '
                            f'{stats[database_name][query_type][ave_recall_name][4]:.2f}   '
                            f'{stats[database_name][query_type][ave_one_percent_recall_name]:.2f}   ')
    wandb.log(data={val_eval: metrics}, step=epoch)
    # if not args.debug:
    #     logger.info(json.dumps(stats))
    torch.cuda.empty_cache()

    # assume the database name is only one
    return metrics['ave_one_percent_recall'][database_name] 

def get_evaluate_without_wandb(model, device, cfgs, data_path, logger, epoch, val_eval, out_put_pair_idxs, wandb_id, model_pc=None):
    model.eval()
    model.to(device)
    with torch.no_grad():
        if cfgs['evaluator_type'] == 1:
            neighbor_num = 25
            stats = evaluate_1(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 2:
            neighbor_num = 500
            stats = evaluate_2(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 3:
            neighbor_num = 25
            stats = evaluate_3(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 4:
            neighbor_num = 25
            stats = evaluate_4(model, device, cfgs, data_path, val_eval, out_put_pair_idxs, wandb_id)
        elif cfgs['evaluator_type'] == 5:
            neighbor_num = 60
            stats = evaluate_5(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 6:
            neighbor_num = 25
            stats = evaluate_6(model, device, cfgs, data_path, val_eval, out_put_pair_idxs, wandb_id)
        elif cfgs['evaluator_type'] == 7:
            neighbor_num = 25
            stats = evaluate_7(model, model_pc, device, cfgs, data_path, val_eval, out_put_pair_idxs, wandb_id)
        elif cfgs['evaluator_type'] == 8:
            neighbor_num = 60
            stats = evaluate_8(model, device, cfgs, data_path, val_eval)
        elif cfgs['evaluator_type'] == 9:
            neighbor_num = 60
            stats = evaluate_9(model, device, cfgs, data_path, val_eval)
        else:
            raise ValueError('Invalid evaluator type: {}'.format(cfgs['evaluator_type']))
    
    logger.info(f'final model of epoch {epoch}:')

    metrics = {}
    metrics['ave_one_percent_recall'] = {}
    metrics['average_sim'] = {}
    metrics['ave_recall'] = {}
    for i in range(neighbor_num):
        metrics['ave_recall'][str(i)] = {}

    for database_name in stats:
        logger.info('Dataset: {}'.format(database_name))
        metrics['ave_one_percent_recall'][database_name] = 0.0
        metrics['average_sim'][database_name] = 0.0
        for i, _ in metrics['ave_recall'].items():
            metrics['ave_recall'][i][database_name] = 0.0
        query_type_num = 0.0
        for query_type in stats[database_name]:
            logger.info('current way of query to database: {}'.format(query_type))
            t = 'Avg. top 1% recall: {:.2f}   Avg. similarity: {:.4f}   Avg. recall @N:'
            logger.info(t.format(stats[database_name][query_type]['ave_one_percent_recall'],
                                    stats[database_name][query_type]['average_sim']))
            logger.info(stats[database_name][query_type]['ave_recall'])
            metrics['ave_one_percent_recall'][database_name] += float(stats[database_name][query_type]['ave_one_percent_recall'])
            metrics['average_sim'][database_name] += float(stats[database_name][query_type]['average_sim'])
            for i, v in enumerate(stats[database_name][query_type]['ave_recall']):
                metrics['ave_recall'][str(i)][database_name] += float(v)
            query_type_num += 1.0
        metrics['ave_one_percent_recall'][database_name] /= max(1.0, query_type_num)
        metrics['average_sim'][database_name] /= max(1.0, query_type_num)
        for i, _ in metrics['ave_recall'].items():
            metrics['ave_recall'][i][database_name] /= max(1.0, query_type_num)

    if cfgs['evaluator_type'] == 5 or cfgs['evaluator_type'] == 8 or cfgs['evaluator_type'] == 9:
        ave_recall_name = 'ave_recall'
        ave_one_percent_recall_name = 'ave_one_percent_recall'
        for database_name in stats:
            for query_type in stats[database_name]:
                logger.info(f'{stats[database_name][query_type][ave_recall_name][0]:.2f}   '
                            f'{stats[database_name][query_type][ave_recall_name][4]:.2f}   '
                            f'{stats[database_name][query_type][ave_one_percent_recall_name]:.2f}   ')
        for database_name in stats:
            for query_type in stats[database_name]:
                print(f'{stats[database_name][query_type][ave_recall_name][0]:.2f}   '
                            f'{stats[database_name][query_type][ave_recall_name][4]:.2f}   '
                            f'{stats[database_name][query_type][ave_one_percent_recall_name]:.2f}   ')