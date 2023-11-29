import numpy as np
import gudhi
import matplotlib.pyplot as plt

def getPersistence(vec, clean=True):
  """
  This function transforms the 1D time series vec into a persistence diagram.
  """
  simplex_up = gudhi.SimplexTree()
  simplex_dw = gudhi.SimplexTree()
  # Fullfill the simplexes
  for i in np.arange(len(vec)): 
    simplex_up.insert([i], filtration=vec[i])
    simplex_dw.insert([i], filtration=-vec[i])
  for i in np.arange(len(vec)-1): 
    simplex_up.insert([i, i+1], filtration=vec[i])
    simplex_dw.insert([i, i+1], filtration=-vec[i])
  # Initialize the filtrations
  simplex_up.initialize_filtration()
  simplex_dw.initialize_filtration()


  dig_up = simplex_up.persistence()
  dig_dw = simplex_dw.persistence()

  if clean:
    dig_up = np.asarray([[ele[1][0], ele[1][1]] for ele in dig_up if ele[1][1] < np.inf])
    dig_dw = np.asarray([[ele[1][0], ele[1][1]] for ele in dig_dw if ele[1][1] < np.inf])

  return dig_up, dig_dw, simplex_up, simplex_dw


def landscapes_approx(diag_dim,x_min,x_max,nb_steps,nb_landscapes):
  """
  Returns nb_landscapes. Same as gudhi implementation
  """
  landscape = np.zeros((nb_landscapes,nb_steps))
  step = (x_max - x_min) / nb_steps
  for i in range(nb_steps):
    x = x_min + i * step
    event_list = []
    for pair in diag_dim:
      b = pair[0]
      d = pair[1]
      if (b <= x) and (x<= d):
        if x >= (d+b)/2. :
          event_list.append((d-x))
        else:
          event_list.append((x-b))
    event_list.sort(reverse=True)
    event_list = np.asarray(event_list)
    for j in range(nb_landscapes):
      if(j<len(event_list)):
        landscape[j,i]=event_list[j]
        
  return landscape

def get_points_landscape(lst, resolution):
    '''
    Given a list corresponding to the values of a landscape
    returns the values of the landscape of a takeoff vertex
    or a local max/min
    '''
    crit = []
    clist = -1
    xx = np.linspace(0, 1, resolution)

    if np.abs(lst[0]) > 10e-6 or (np.abs(lst[0]) < 10e-6 and np.abs(lst[1]) > 10e-6):
        clist += 1
        crit.append(xx[0])

    for i in range(1, len(lst)-1):
        #TAKE-OFF VERTEX
        # first value such that it is greater than 0 corresponds to a b coordinate
        if np.abs(lst[i-1]) < 10e-6 and np.abs(lst[i]) < 10e-6 and np.abs(lst[i+1]) > 10e-6:
            crit.append(xx[i])
            clist += 1
        #LOCAL MAX OR MIN
        # local maximums and minimums need to change coordinates
        elif (lst[i-1] < lst[i] and lst[i+1] < lst[i]) or (lst[i-1] > lst[i] and lst[i+1] > lst[i]):
            crit.append(xx[i])
            clist += 1
            crit[clist] = 2 * (crit[clist] - 0.5*crit[clist-1])
    if np.abs(lst[len(lst)-1]) > 10e-6:
        crit.append(xx[len(lst)-1])
        clist += 1
    return crit

def get_index_points_landscape(lst):
    '''
    Given a list corresponding to the values of a landscape
    returns a list of indices of the landscape where a takeoff vertex
    or a local max/min occurrs
    '''
    crit = []

    if np.abs(lst[0]) > 10e-6 or (np.abs(lst[0]) < 10e-6 and np.abs(lst[1]) > 10e-6):
        crit.append(0)

    for i in range(1, len(lst)-1):
        #TAKE-OFF VERTEX
        # first value that is greater than 0 corresponds to a b coordinate
        if np.abs(lst[i-1]) < 10e-6 and np.abs(lst[i]) < 10e-6 and np.abs(lst[i+1]) > 10e-6:
            crit.append(i)
        #LOCAL MAX OR MIN
        # local maximums and minimums need to change coordinates
        elif (lst[i-1] < lst[i] and lst[i+1] < lst[i]) or (lst[i-1] > lst[i] and lst[i+1] > lst[i]):
            crit.append(i)

    if np.abs(lst[len(lst)-1]) > 10e-6:
        crit.append(len(lst)-1)

    return crit

def get_closest_points(lst_crit, original):
    '''
    Given a list of critical y-values, returns the x-values
    of the original function that are actually critical points
    '''
    x_crit = []
    for m in lst_crit:
        temp = (original-m)**2
        temp = temp<10e-6
        temp_list = [i for i, x in enumerate(temp) if x]

    
        final_list= []
        for i in temp_list:
            if i == 0 or i == len(original)-1:
                final_list.append(i)
            elif (original[i] < original[i-1] and original[i] < original[i+1]) or \
            (original[i] > original[i-1] and original[i] > original[i+1]) or \
            (np.abs(original[i-1]-original[i]) < 10e-8) or \
            (np.abs(original[i+1]-original[i]) < 10e-8):
                final_list.append(i)

        x_crit = x_crit + final_list
    return x_crit

def reconstruct_function(landscapes, original, res, levels=4):
    '''
    Given a list of landscapes, reconstruct the original function
    '''
    aa = []
    for i in range(levels):
        crit = get_points_landscape(landscapes[i], res)
        aa = aa + crit

    xx = get_closest_points(aa, original)
    xx =list(set(xx))
    idx = np.argsort(xx)
    xx = [xx[i] for i in idx]
    yy = [original[xi] for xi in xx]

    x_real = []
    y_real = []
    for i in range(len(xx)-2):
        x_real.append(xx[i])
        y_real.append(yy[i])
        if np.abs(yy[i]) < 10e-8 and np.abs(yy[i+1]) < 10e-8:
            break
    x_real.append(xx[len(xx)-1])
    y_real.append(yy[len(yy)-1])

    return x_real, y_real

def reconstruct_function_modified(landscapes, original, res, levels=4):
    aa = []
    step = []
    for i in range(levels):
        crit = get_points_landscape(landscapes[i], res)
        aa = aa + crit
        if i == levels-1:
            step = crit

    xx = get_closest_points(aa, original)
    step_real = get_closest_points(step, original)

    xx =list(set(xx))
    step_real = list(set(step_real))

    idx = np.argsort(xx)
    xx = [xx[i] for i in idx]
    yy = [original[xi] for xi in xx]

    idx_step = np.argsort(step_real)
    xx_step = [step_real[i] for i in idx_step]
    yy_step = [original[xi] for xi in xx_step]


    return xx, yy, xx_step, yy_step

def plot_paralel(LL, resolution, ynew, xnew, step):
    fig = plt.figure(figsize=(25, 7))

    ax = fig.add_subplot(121)
    xch = get_index_points_landscape(LL[step])
    ych = np.linspace(0, 1, num=resolution, endpoint=True)
    ax.plot(ych, LL[step])
    ax.plot([ych[i] for i in xch], [LL[step][i] for i in xch], "o", color='red')
    ax.title.set_text(str(step)+'-th Landscape')
    ax.set_ylim(-0.05, 1)

    ax2 = fig.add_subplot(122)
    xx, yy, xx_step, yy_step = reconstruct_function_modified(LL, ynew, resolution, step+1)
    ax2.plot(xnew, ynew)
    ax2.plot([xnew[i] for i in xx], yy, "*-")
    ax2.plot([xnew[i] for i in xx_step], yy_step, "o", color='red')
    ax2.title.set_text('Reconstructing original function')


    plt.show()