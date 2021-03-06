import argparse

from src import file_loader, paths, functions

CROSS_VALIDATION_FOLD = 10


def _init_parser():
    """
    Parser for command line arguments for the program
    """

    parser = argparse.ArgumentParser(
        description="Decision tree training on "
        "clean and noisy data sets. "
        "With pruning capabilities",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c", "--clean", action="store_true", help="cross-validation experiment on clean data set"
    )

    parser.add_argument(
        "-n", "--noisy", action="store_true", help="cross-validation experiment on noisy data set"
    )

    parser.add_argument(
        "-p", "--prune", action="store_true", help="perform operation with pruning on the tree"
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="run full experiment with cross-validation on "
        "clean and noisy data sets then repeat with "
        "pruning",
    )

    parser.add_argument(
        "-sc",
        "--showc",
        action="store_true",
        help="train a single decision tree on the clean data " "set, plotting out the tree",
    )

    parser.add_argument(
        "-sn",
        "--shown",
        action="store_true",
        help="train a single decision tree on the noisy data " "set, plotting out the tree",
    )

    parser.add_argument(
        "FOLD", metavar="fold", type=int, help="number of folds to use in cross validation"
    )

    args = parser.parse_args()
    return args


def main():
    """
    Entry point for program, call other functions here
    """

    # parse command line arguments
    args = _init_parser()

    # load data sets
    clean_data = file_loader.load_data(paths.CLEAN_DATA)
    noisy_data = file_loader.load_data(paths.NOISY_DATA)

    if args.all or args.clean:
        print("Evaluating trees on the clean dataset")
        functions.cross_validate(clean_data, args.FOLD)
        print()
    if args.all or args.noisy:
        print("Evaluating trees on the noisy dataset")
        functions.cross_validate(noisy_data, args.FOLD)
        print()
    if args.all or (args.prune and args.clean):
        print("Evaluating trees on clean data set with pruning")
        functions.cross_validate_and_prune(clean_data, args.FOLD)
        print()
    if args.all or (args.prune and args.noisy):
        print("Evaluating trees on noisy data set with pruning")
        functions.cross_validate_and_prune(noisy_data, args.FOLD)
    if args.showc:
        print(
            "Training and Plotting a tree on clean data {} pruning".format(
                "with" if args.prune else "without"
            )
        )
        functions.train_and_plot_tree(clean_data, args.FOLD, args.prune)
    if args.shown:
        print(
            "Training and Plotting a tree on noisy data {} pruning".format(
                "with" if args.prune else "without"
            )
        )
        functions.train_and_plot_tree(noisy_data, args.FOLD, args.prune)


if __name__ == "__main__":
    main()
