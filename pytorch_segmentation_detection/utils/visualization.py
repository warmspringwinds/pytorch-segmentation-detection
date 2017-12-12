
class VizList(list):
    """Extended List class which can be binded to an matplotlib's pyplot axis
    and, when being appended a value, automatically update the figure.
    
    Originally designed to be used in a jupyter notebook with activated
    %matplotlib notebook mode.
    
    Example of usage:
    
    %matplotlib notebook

    from matplotlib import pyplot as plt

    f, (loss_axis, validation_axis) = plt.subplots(2, 1)
    loss_axis.set_title('Training loss')
    validation_axis.set_title('MIoU on validation dataset')
    plt.tight_layout()
    
    loss_list = VizList()

    validation_accuracy_res = VizList()
    train_accuracy_res = VizList()

    loss_axis.plot([], [])

    validation_axis.plot([], [], 'b',
                         [], [], 'r')

    loss_list.bind_to_axis(loss_axis)

    validation_accuracy_res.bind_to_axis(validation_axis, 0)
    train_accuracy_res.bind_to_axis(validation_axis, 1)
    
    Now everytime the list are updated, the figure are updated
    automatically:
    
    # Run multiple times
    loss_list.append(1)
    loss_list.append(2)
    
    
    Attributes
    ----------
    axis : pyplot axis object
        Axis object that is being binded with a list
    axis_index : int
        Index of the plot in the axis object to bind to
        
    """
    
    def __init__(self, *args):
        
        super(VizList, self).__init__(*args)
        
        self.object_count = 0
        self.object_count_history = []
        
        self.axis = None
        self.axis_index = None
        
    def append(self, object):
        
        self.object_count += 1
        self.object_count_history.append(self.object_count)
        super(VizList, self).append(object)
        
        self.update_axis()
    
    def bind_to_axis(self, axis, axis_index=0):
        
        self.axis = axis
        self.axis_index = axis_index
    
    def update_axis(self):
        
        self.axis.lines[self.axis_index].set_xdata(self.object_count_history)
        self.axis.lines[self.axis_index].set_ydata(self)
        
        self.axis.relim()
        self.axis.autoscale_view()
        self.axis.figure.canvas.draw()