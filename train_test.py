import torch
import utility
import CONSTANTS
from tqdm import tqdm
import torch.nn.functional as f


def train_test(*, model, device, train_loader, optimizer, epochs, scheduler, test, test_loader, type_, tracker, loss_fn,
               l1_lambda=None, l2_lambda=None):
    if CONSTANTS.L1 in utility.get_config_details()[CONSTANTS.REGULARIZATION].keys():
        l1_lambda = float(utility.get_config_details()[CONSTANTS.REGULARIZATION][CONSTANTS.L1]),
    if CONSTANTS.L2 in utility.get_config_details()[CONSTANTS.REGULARIZATION].keys():
        l2_lambda = float(utility.get_config_details()[CONSTANTS.REGULARIZATION][CONSTANTS.L2])

    model.train()

    for epoch in range(epochs):
        l1 = torch.tensor(0, requires_grad=False)
        correct = 0
        processed = 0
        train_loss = 0

        print(f"\n\nepoch: {epoch + 1}")
        # pbar = tqdm(train_loader)

        if "l2" in type_.lower():
            optimizer.param_groups[0]['weight_decay'] = l2_lambda

        for i, data in enumerate(train_loader, 0):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            optimizer.zero_grad()  # set the gradients top zero to avoid accumulatin them over the epochs

            output = model(inputs)  # model's output

            loss = loss_fn(output, target)

            # if "l1" in type_.lower():
            #     for param in model.parameters():
            #         l1 = l1 + param.abs().sum()
            #     loss = loss + l1_lambda * l1.item()

            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            train_loss += loss.item()
            if i > 0 and i % 500 == 0:
                print(f"epoch:{epoch + 1}.... batch:{i + 1}...loss:{train_loss:.4f}")

        train_loss /= len(train_loader.dataset)
        acc = 100 * correct / processed
        tracker[type_]['train_losses'].append(train_loss)
        tracker[type_]['train_accuracy'].append(acc)

        # pbar.set_description(desc=f'loss={loss.item()} batch_id={batch_idx}')
        if scheduler:
            print(f'\n>>>lr: {scheduler.get_last_lr()[0]}')
            scheduler.step()
        print(f"\nEpoch : {epoch+1} Train loss:{train_loss:.4f}")
        print("\nTrain set: \t\t Accuracy: {}/{} ({:.6f}%)".format(correct, len(train_loader.dataset),
                                                                   100.0 * correct / len(train_loader.dataset)))

        if test:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += loss_fn(output, target).sum().item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    for i in range(len(pred)):
                        if pred[i] != target[i]:
                            tracker[type_]['misclassified'].append((data[i], pred[i], target[i]))

            test_loss /= len(test_loader.dataset)
            t_acc = 100.0 * correct / len(test_loader.dataset)
            tracker[type_]['test_losses'].append(test_loss)
            tracker[type_]['test_accuracy'].append(t_acc)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), t_acc))
