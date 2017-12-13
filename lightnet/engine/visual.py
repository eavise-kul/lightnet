#
#   Visualisation with visdom
#   Copyright EAVISE
#   

import visdom
import numpy as np
import brambox.boxes as bbb

from ..logger import *

__all__ = ['Visualisation']


class Visualisation:
    """ This class contains visualisation functions,
    that can automatically compute and plot certain statistics for you.

    Args:
        options (dict): Options that are passed on to the Visdom constructor
    """
    def __init__(self, options):
        self.vis = visdom.Visdom(**options)
        if not self.vis.check_connection():
            log(Loglvl.ERROR, f'Could not connect to visdom server', OSError)

        if 'env' in options:
            self.vis.close(env=options['env'])

    def pr(self, pr, window=None, **options):
        """ Plot Precision and Recall curves.
        
        Args:
            pr (dict): Dictionary containing (p, r) tuples (eg. output of brambox.boxes.pr)
            window (str, optional): Name of the visdom window
            **options (dict): Extra options to pass to the Visdom.line function
        """
        x = [val[1] for key, val in pr.items()]
        y = [val[0] for key, val in pr.items()]
        legend = [f'{key}: {round(bbb.ap(*val)*100, 2)}' for key, val in pr.items()]

        opts = dict(
            xlabel='Recall',
            ylabel='Precision',
            legend=legend,
            xtickmin=0,
            xtickmax=1,
            ytickmin=0,
            ytickmax=1,
            **options
                )
        self.vis.line(X=np.array(x).transpose(), Y=np.array(y).transpose(), win=window, opts=opts)

    def loss(self, loss, batch, window, name=None, **options):
        """ Plot loss curve.
        
        Args:
            loss (Number): new loss value
            batch (Number): batch number
            window (str): Name of the visdom window
            name (str, optional): Name of the visdom line inside the window
            **options (dict): Extra options to pass to the Visdom.line function
        """
        x = np.array([batch])
        y = np.array([loss])
        
        if not 'legend' in options and name is not None:
            options['legend'] = [name]
            
        if not self.vis.win_exists(window):
            opts = dict(
                xlabel='Batch',
                ylabel='Loss',
                **options
                )
            update = None
        else:
            opts = options
            update = 'append'

        self.vis.line(X=x, Y=y, win=window, name=name, update=update, opts=opts)
