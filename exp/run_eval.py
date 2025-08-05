from lib.kits.eval import *


def eval_pipeline(pipeline, data):
    evaluator = UniformEvaluator(data, pipeline.device)

    for batch in tqdm(data['data_loader']):
        batch = recursive_to(batch, pipeline.device)

        with torch.no_grad():
            #out = pipeline(batch['img_patch'])
            out = pipeline(batch)

        evaluator.eval(pd=out, gt=batch)

    return evaluator.get_results()


if __name__ == '__main__':
    args = parse_args()
    exp_root = Path(args.exp_root).absolute()

    # 1. Load exp config.
    cfg_path = exp_root / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(cfg_path)

    # 2. Load data.
    get_logger(brief=True).info(f'Will test on {args.dataset}')
    data = get_data(args.dataset, cfg)

    # 3. Prepare the pipeline.
    pipeline = instantiate(cfg.pipeline, init_backbone=False, _recursive_=False).to(args.device)
    get_logger(brief=True).info(f'Pipeline initialized.')
    ckpt_path = exp_root / 'last.ckpt'  #to modify
    pipeline.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    pipeline.eval()

    # 4. Read cached results.
    results_path = PM.outputs / 'evaluation' / f'{cfg.exp_name}-standard_benchmark.npy'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if results_path.exists():
        results_all_ds = np.load(results_path, allow_pickle=True).item()
    else:
        results_all_ds = {}

    # 5. Evaluation.
    results = eval_pipeline(pipeline, data)
    rich.print(f'{args.dataset}:\n{results}')

    # 6. Dump the results.
    results_all_ds.update({args.dataset: results})
    np.save(results_path, results_all_ds)