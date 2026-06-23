from egno_eval_visualize import main


if __name__ == "__main__":
    main(
        default_experiments=["mocap_exp_walk", "mocap_exp_run"],
        default_output_dir="outputs/egno_mocap_eval",
    )
