import traceback
import torch.nn as nn
import CONSTANTS, train_test, train_test_dataloader, utility
from models import basic_mnist, cifar10_groups_dws_s7_model, resnet_model


def run_model_run(dataset=None, *, session="s8"):
    try:
        train_transforms, test_transforms = train_test_dataloader.define_train_test_transformers(session=session)
        train_data, test_data = train_test_dataloader.download_data(
            dataset_name=utility.get_dataset_name(session=session),
            train_transforms=train_transforms,
            test_transforms=test_transforms)

        train_loader, test_loader = train_test_dataloader.get_train_test_dataloaders(train_data=train_data,
                                                                                     test_data=test_data,
                                                                                     data_loader_args=utility.get_dataloader_args())

        all_regularizations_list, tracker = utility.get_combos_and_trackers()
        device = utility.get_device()
        # utility.get_all_models_summary()
        loss_fn = nn.functional.nll_loss
        model = None

        for combo in all_regularizations_list:
            print("\nRunning for: ", combo)

            if dataset and dataset.lower() == "mnist":
                if CONSTANTS.GBN in combo.lower():
                    model = basic_mnist.GBNNet().to(device)
                else:
                    model = basic_mnist.S6_MNIST().to(device)
            elif "s8" in session.lower() or dataset.lower() == "cifar10":
                #model = cifar10_groups_dws_s7_model.S7_CIFAR10()
                model = resnet_model.BasicBlock()
                model = model.to(device)
                loss_fn = nn.CrossEntropyLoss()

            optimizer = utility.get_optimizer(model=model)
            scheduler = utility.get_scheduler(optimizer=optimizer)
            utility.show_model_summary(title=model.__doc__, model=model, input_size=utility.get_input_size(
                dataset=utility.get_dataset_name(session=session)))

            train_test.train_test(model=model, device=device, train_loader=train_loader, optimizer=optimizer,
                                  epochs=int(utility.get_config_details()[CONSTANTS.MODEL_CONFIG][CONSTANTS.EPOCHS]),
                                  scheduler=scheduler,
                                  test=True, test_loader=test_loader, type_=combo, tracker=tracker,
                                  loss_fn=loss_fn)

        for plot_type in utility.get_config_details()[CONSTANTS.PLOTS][CONSTANTS.TO_PLOT].strip().split(','):
            utility.plot(title="Plot is for:" + plot_type, x_label='Epochs', y_label=plot_type.lower(),
                         tracker=tracker, category=plot_type)
    except Exception as e:
        print(traceback.format_exc(e))


if __name__ == '__main__':
    run_model_run(session="s7")
