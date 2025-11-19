from tensorflow import keras
import time

class ASK(keras.callbacks.Callback):
    def __init__(self, model, epochs, ask_epoch):
        super(ASK, self).__init__()
        self.model = model
        self.ask_epoch = ask_epoch
        self.epochs = epochs
        self.ask = True

    def on_train_begin(self, logs=None):
        if self.ask_epoch == 0:
            print('you set ask_epoch = 0, ask_epoch will be set to 1', flush=True)
            self.ask_epoch = 1
        if self.ask_epoch >= self.epochs:
            print('ask_epoch >= epochs, will train for ', self.epochs, ' epochs', flush=True)
            self.ask = False
        if self.epochs == 1:
            self.ask = False
        else:
            print('Training will proceed until epoch', self.ask_epoch, ' then you will be asked to')
            print(' enter H to halt training or enter an integer for how many more epochs to run then be asked again')
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        tr_duration = time.time() - self.start_time
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print(msg, flush=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.ask:
            if epoch + 1 == self.ask_epoch:
                print('\n Enter H to end training or an integer for the number of additional epochs to run then ask again')
                ans = input()
                if ans == 'H' or ans == 'h' or ans == '0':
                    print('you entered ', ans, ' Training halted on epoch ', epoch+1, ' due to user input\n', flush=True)
                    self.model.stop_training = True
                else:
                    self.ask_epoch += int(ans)
                    if self.ask_epoch > self.epochs:
                        print('\nYou specified maximum epochs as ', self.epochs, ' cannot train for ', self.ask_epoch, flush=True)
                    else:
                        print('you entered ', ans, ' Training will continue to epoch ', self.ask_epoch, flush=True)