
def linear_fit(x,y):

	r""" Ordinary linear least regressions fit to a 1d array of x and y data

	Parameters
	----------
	x : numpy array
		the independent variable
	y : numpy array
		the dependent variable 

	Returns
	-------
	opts : ``LinregressResult`` instance
		The return value is an object with the following attributes

		slope : float
			Slope of the regression line.
		intercept : float
			Intercept of the regression line.
		rvalue : float
			The Pearson correlation coefficient. The square of ``rvalue``
			is equal to the coefficient of determination.
		pvalue : float
			The p-value for a hypothesis test whose null hypothesis is
			that the slope is zero, using Wald Test with t-distribution of
			the test statistic. 
		stderr : float
			Standard error of the estimated slope (gradient), under the
			assumption of residual normality.
		intercept_stderr : float
			Standard error of the estimated intercept, under the assumption
			of residual normality.

	"""
	from scipy.stats import linregress
	
	opts = linregress(x,y)
	
	return opts


def exponential_decay_correction_time(signal, time_subset=None, f_scale=100., loss_type='soft_l1'):
	
	r""" Corrects the mean intensities of an input video by fitting an exponential decay curve. The exponential is of the mathematical form

	.. math::
		y = Ae^{Bt} + C

	where A, B, C are constants, t is time and y the signal. 
	Robust regression is used for fitting. See https://scipy-cookbook.readthedocs.io/items/robust_regression.html for more information.

	Parameters
	----------
	signal : 1D numpy array
		The 1D signal to fit an exponential function. 
	time_subset :  1D numpy array
		A 1D array to mask the fitting to specific contiguous timepoints
	f_scale : float
		The regularisation parameter in the robust fitting
	loss_type: str
		One of the 5 robust loss types available in scipy, https://scipy-cookbook.readthedocs.io/items/robust_regression.html

	Returns
	-------
	correction_factor : 1D numpy array
		the multiplication factor at each timepoint to correct the signal
	(res_robust.x, robust_y) : 2-tuple
		res_robust.x returns the fitted model coefficients for reuse, 
		robust_y is the prediction of the signal using the fitted model coefficients 

	See Also
	--------
	unwrap3D.Analysis_Functions.timeseries.baseline_correction_time : 
		Use weighted least squares to estimate a baseline signal that can be used for correction

	"""
	from scipy.stats import linregress
	from scipy.optimize import least_squares
	import numpy as np 
	
	I_vid = signal.copy()
	
	if time_subset is None:
		# use the full
		I_time = np.arange(len(I_vid))
		I_time_subset = I_time.copy()
		I_vid_subset = I_vid.copy()
	else:
		I_time = np.arange(len(I_vid))
		I_time_subset = I_time[time_subset].copy()
		I_vid_subset = I_vid[time_subset].copy()
		
	# fit equation. y =A*e^(-Bt)
	log_I_vid = np.log(I_vid_subset)
	slope, intercept, r_value, p_value, std_err = linregress(I_time_subset, log_I_vid)

	# initial fit. 
	A = np.exp(intercept)
	B = slope
	# refined robust fitting. 
	def exp_decay(t,x):
		return (x[0] * np.exp(x[1] * t) + x[2])
	
	# def exp_decay_line(t,x):    
		# return (x[0] * np.exp(x[1] * (t+x[3])) + x[2]) # exp + linear. (rather than this.... might be more correct to be a switch.... )
	def res(x, t, y):
		# return exp_decay(t,x) - y
		return exp_decay(t,x) - y 
	
	x0 = [A, B, 0] #, 0]
	# print(x0)
	res_robust = least_squares(res, x0, loss=loss_type, f_scale=f_scale, args=(I_time_subset, I_vid_subset))
		
	"""
	applying the fitted now on the proper sequence. 
	"""
	robust_y = exp_decay(I_time, res_robust.x)
	correction_factor = float(robust_y[0]) / robust_y
	
	return correction_factor, (res_robust.x, robust_y)


def baseline_correction_time(signal, p=0.1, lam=1, niter=10):
	r""" Corrects the mean signal using weighted least square means regression to infer a baseline signal. 
	Specifically we estimate the baseline signal, :math:`z` as the solution to the following optimization problem 

	.. math::
		z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

	where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

	.. math::
		w = 
		\Biggl \lbrace 
		{ 
		p ,\text{ if } 
		  {y>z}
		\atop 
		1-p, \text{ otherwise } 
		}

	Parameters
	----------
	signal : 1D numpy array
		The 1D signal to estimate a baseline signal. 
	p :  scalar
		Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
	lam : scalar
		Controls the degree of smoothness in the baseline
	niter: int
		The number of iterations to run the algorithm. Only a few iterations is required generally. 

	Returns
	-------
	correction_factor : 1D numpy array
		the multiplication factor at each timepoint to correct the signal
	(baseline, corrected) : 2-tuple of 1D numpy array 
		baseline is the inferred smooth baseline signal, 
		corrected is the multiplicatively corrected signal using the estimated correction factor from the baseline

	See Also
	--------
	unwrap3D.Analysis_Functions.timeseries.baseline_als : 
		The weighted least squares method used to estimate the baseline

	"""
	from scipy.stats import linregress
	from scipy.optimize import least_squares
	import numpy as np 
	
	I_vid = signal
	baseline = baseline_als(I_vid, lam=lam, p=p, niter=niter)

	correction_factor = float(I_vid[0]) / baseline
	corrected = I_vid * correction_factor 
	
	return correction_factor, (baseline, corrected)


def baseline_als(y, lam, p, niter=10):
	r""" Estimates a baseline signal using asymmetric least squares. It can also be used for generic applications where a 1D signal requires smoothing.
	Specifically the baseline signal, :math:`z` is the solution to the following optimization problem 

	.. math::
		z = arg\,min_z \{w(y-z)^2 + \lambda\sum(\Delta z)^2\}

	where :math:`y` is the input signal, :math:`\Delta z` is the 2nd derivative, :math:`\lambda` is the smoothing regularizer and :math:`w` is an asymmetric weighting

	.. math::
		w = 
		\Biggl \lbrace 
		{ 
		p ,\text{ if } 
		  {y>z}
		\atop 
		1-p, \text{ otherwise } 
		}

	Parameters
	----------
	signal : 1D numpy array
		The 1D signal to estimate a baseline signal. 
	p :  scalar
		Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
	lam : scalar
		Controls the degree of smoothness in the baseline
	niter: int
		The number of iterations to run the algorithm. Only a few iterations is required generally. 

	Returns
	-------
	z : 1D numpy array
		the estimated 1D baseline signal

	See Also
	--------
	unwrap3D.Analysis_Functions.timeseries.baseline_correction_time : 
		Application of this method to estimate a baseline for a 1D signal and correct the signal e.g. for photobleaching
	unwrap3D.Analysis_Functions.timeseries.decompose_nonlinear_time_series :
		Application of this method to decompose a 1D signal into smooth baseline + high frequency fluctuations. 

	"""
	from scipy import sparse
	from scipy.sparse.linalg import spsolve
	import numpy as np 
	
	L = len(y)
	D = sparse.csc_matrix(np.diff(np.eye(L), 2))
	w = np.ones(L)
	for i in range(niter):
		W = sparse.spdiags(w, 0, L, L)
		Z = W + lam * D.dot(D.transpose())
		z = spsolve(Z, w*y)
		w = p * (y > z) + (1-p) * (y < z)
	return z


def fit_spline(x,y, smoothing=None):
    r""" Fitting a univariate smoothing spline to x vs y univariate data curve

    Parameters
    ----------
    x : (N,) array
        the 1d x- variables
    y : (N,) array
        the 1d y- variables
    smoothing : scalar
        Optional control of smoothing, higher gives more smoothing. If None, it is automatically set based on standard deviation of y.

    Returns
    -------
    x, y_norm, interp_s : list of (N,) arrays
        the 1d x- variables, normalised maximum normalized y_norm=y/y_max variables used to fit and interp_s the smooth y curve  
    spl : scipy spline instance object
        the interpolating spline object fitted on x, y_norm   

    """    
    from scipy.interpolate import UnivariateSpline
    import numpy as np 
    
    max_y = np.max(y)
    y_norm = y/float(max_y)   
    
    if smoothing is None:
        spl = UnivariateSpline(x, y_norm, s=2*np.std(y_))
    else:
        spl = UnivariateSpline(x, y_norm, s=smoothing)
        
    interp_s = max_y*spl(x)
    
    return (x, y_norm, interp_s), spl

def decompose_nonlinear_time_series(y, lam, p, niter=10, padding=None):
	r""" Decomposes a given signal into smooth + residual components. The smooth signal is estimated using asymmetric least squares.  

	Parameters
	----------
	y : 1D numpy array
		The input 1D signal to decompose 
	lam : scalar
		Controls the degree of smoothness in the baseline
	p :  scalar
		Controls the degree of asymmetry in the weighting. p=0.5 is the same as smoothness regularized least mean squares.
	niter: int
		The number of iterations to run the algorithm. Only a few iterations is required generally. 
	padding: int
		Padding window to dampen boundary conditions. If None, the entire signal is used for padding otherwise the specified window is used to pad. 
		Reflecting boundary conditions are used for padding.   

	Returns
	-------
	z : 1D numpy array
		the estimated 1D baseline signal
	"""
	import numpy as np 
	
	if padding is None:
		y_ = np.hstack([y[::-1], y, y[::-1]])
		y_base = baseline_als(y_, lam=lam, p=p, niter=niter)
		y_base = y_base[len(y):-len(y)]
	else:
		y_ = np.hstack([y[::-1][-padding:], y, y[::-1][:padding]]) 
		y_base = baseline_als(y_, lam=lam, p=p, niter=niter)
		y_base = y_base[padding:-padding]
		
	return y_base, y-y_base
	

def xcorr(x, y=None, norm=True, eps=1e-12):
	r""" Computes the discrete crosscorrelation of two read 1D signals as defined by

	.. math::
		c_k = \sum_n x_{n+k} \cdot y_n

	If norm=True, :math:`\hat{x}=\frac{x-\mu}{\sigma}` and :math:`\hat{y}=\frac{y-\mu}{\sigma}` is normalised and the zero-normalised autocorrelation is computed,  
	
	.. math::
		c_k = \frac{1}{T}\sum_n \hat{x}_{n+k} \cdot \hat{y}_n

	where :math:`T` is the length of the signal :math:`x`

	Parameters
	----------
	x : 1D numpy array
		The input 1D signal
	y : 1D numpy array
		The optional second 1D signal. If None, the autocorrelation of x is computed.   
	norm : bool
		If true, the normalized autocorrelation is computed such that all values are in the range [-1.,1.]
	eps :  scalar
		small constant to prevent zero division when norm=True

	Returns
	-------
	result : 1D numpy array
		the 1-sided autocorrelation if y=None, or the full cross-correlation otherwise

	Notes
	-----
	The definition of correlation above is not unique and sometimes correlation
	may be defined differently. Another common definition is:

	.. math:: 
		c'_k = \sum_n x_{n} \cdot {y_{n+k}}
	
	which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.
	"""
	import numpy as np 

	if norm: 
		a = (x - np.nanmean(x)) / (np.nanstd(x) * len(x) + eps)
		if y is not None:
			b = (y - np.nanmean(y)) / (np.nanstd(y) + eps)
		else:
			b = a.copy()
	else:
		a = x.copy()
		if y is not None:
			b = y.copy()
		b = x.copy()
	result = np.correlate(a, b, mode=mode) # this is not normalized!. 

	if y is None: 
		# return the 1-sided autocorrelation. 
		result = result[result.size // 2:]

	return result

def xcorr_timeseries_set_1d(timeseries_array1, timeseries_array2=None, norm=True, eps=1e-12, stack_final=False):
	r""" Computes the discrete crosscorrelation of two read 1D signals as defined by

	.. math::
		c_k = \sum_n x_{n+k} \cdot y_n

	If norm=True, :math:`\hat{x}=\frac{x-\mu}{\sigma}` and :math:`\hat{y}=\frac{y-\mu}{\sigma}` is normalised and the zero-normalised autocorrelation is computed,  
	
	.. math::
		c_k = \frac{1}{T}\sum_n \hat{x}_{n+k} \cdot \hat{y}_n

	where :math:`T` is the length of the signal :math:`x`

	given two arrays or lists of 1D signals. The signals in the individual arrays need not have the same temporal length. If they do, by setting stack_final=True, the result can be returned as a numpy array else will be returned as a list  

	Parameters
	----------
	timeseries_array1 : array_like of 1D signals 
		A input list of 1D signals 
	timeseries_array2 : array_like of 1D signals 
		An optional second 1D signal set. If None, the autocorrelation of each timeseries in timeseries_array1 is computed.   
	norm : bool
		If true, the normalized autocorrelation is computed such that all values are in the range [-1.,1.]
	eps :  scalar
		small constant to prevent zero division when norm=True
	stack_final : bool
		if timeseries_array1 or timeseries_array2 are numpy arrays with individual signals within of equal temoporal length, setting this flag to True, will return a numpy array else returns a list of cross-correlation curves

	Returns
	-------
	xcorr_out : array_list of 1D numpy array
		the 1-sided autocorrelation if y=None, or the full cross-correlation otherwise

	Notes
	-----
	The definition of correlation above is not unique and sometimes correlation
	may be defined differently. Another common definition is:

	.. math:: 
		c'_k = \sum_n x_{n} \cdot {y_{n+k}}
	
	which is related to :math:`c_k` by :math:`c'_k = c_{-k}`.
	"""
	import numpy as np 
	compute_xcorr=True 
	if timeseries_array2 is None:
		timeseries_array2 = timeseries_array1.copy() # create a copy.
		compute_xcorr=False

	xcorr_out = []
	for ii in np.arange(len(timeseries_array1)):
		timeseries_ii_1 = timeseries_array1[ii].copy()
		timeseries_ii_2 = timeseries_array2[ii].copy()
		if compute_xcorr:
			xcorr_timeseries_ii = xcorr(timeseries_ii_1, y=timeseries_ii_2, norm=norm, eps=eps)
		else:
			xcorr_timeseries_ii = xcorr(timeseries_ii_1, y=None, norm=norm, eps=eps)
		xcorr_out.append(xcorr_timeseries_ii)

	if stack_final:
		xcorr_out = np.vstack(xcorr_out)
		return xcorr_out
	else:
		return xcorr_out
	
def stack_xcorr_curves(xcorr_list):
	r""" Compiles a list of cross-correlation curves of different temporal lengths with NaN representing missing values assuming the midpoint of each curves is time lag = 0

	Parameters
	----------
	xcorr_list : list of 1D arrays
		A input list of 1D cross-correlation curves of different lengths 
	
	Returns
	-------
	out_array : numpy array
		a compiled N x T matrix, where N is the number of curves, and T is the length of the longest curve.

	"""
	import numpy as np 
	
	N = [len(xx) for xx in xcorr_list]
	size = np.max(N)
	out_array = np.zeros((len(xcorr_list), size)); out_array[:] = np.nan
	
	for jj in np.arange(len(xcorr_list)):
		xcorr = xcorr_list[jj].copy()
		out_array[jj,size//2-len(xcorr)//2:size//2+len(xcorr)//2+1] = xcorr.copy()
		
	return out_array
	

def spatialcorr_k_neighbors(timeseries_array, k_graph, norm=True, eps=1e-12):
	r""" Computes the spatial correlation of timeseries for a defined number of steps away. That is it assumes the distance has been pre-discretised and the k_graph represents the relevant neighbors at different distance steps away for a timeseries.

	Parameters
	----------
	timeseries_array : (NxT) of 1D timeseries
		A input list of 1D cross-correlation curves of different lengths 
	k_graph : N x (adjacency list of neighbors)
		adjacency list of each timeseries for N radial spatial steps
	norm : bool
		if True, returns the zero-normalised correlation coefficient in [-1.,1.]
	eps : scalar
		small constant to prevent zero division when norm=True
	
	Returns
	-------
	vertex_means_pearsonr : 1d numpy array
		The average spatial correlation over all timeseries at the given spatial steps away 

	"""

	import numpy as np 
	
	z_norm = timeseries_array.copy()
	if norm:
		z_norm = (z_norm-np.nanmean(z_norm, axis=1)[:,None]) / (np.nanstd(z_norm, axis=1)[:,None]+eps)
	
	adj_list = list(k_graph)
	##### too much memory usage. 
	# N_adj_list = np.hstack([len(aa) for aa in adj_list])
	
	# adj_list_pad = -np.ones((len(N_adj_list), np.max(N_adj_list)), dtype=np.int32)
	# for vv_ii in np.arange(len(adj_list)):
	#     adj_list_pad[vv_ii, :len(adj_list[vv_ii])] = adj_list[vv_ii]
	
	# series1 = z_norm[adj_list_pad].copy()
	# series1_mask = np.ones(adj_list_pad.shape, dtype=bool); series1_mask[adj_list_pad==-1] = 0 
	
	# vertex_means_pearsonr = np.nanmean(z_norm[:,None,:] * series1, axis=-1) 
	# vertex_means_pearsonr = np.nansum(vertex_means_pearsonr*series1_mask, axis=1) / N_adj_list
	
	# all_ring_corrs.append(vertex_means_pearsonr)
	vertex_means_pearsonr = []
	
	# iterate over each vertex.
	for vv_ii in np.arange(len(adj_list)): # this is likely the longest loop - we can paralellize this if we pad.... 
	#     series0 = z_norm[vv_ii].copy() # if norm. 
	#     series1 = z_norm[adj_list[vv_ii]].copy()
		
	#     corrs = np.hstack([spstats.pearsonr(series0, ss)[0] for ss in series1])
	#     vertex_means_pearsonr.append(np.nanmean(corrs))
		
	#     # hard code this to make this fast. 
	# # definition being Sxy / SxxSyy 
	# # https://stackabuse.com/calculating-pearson-correlation-coefficient-in-python-with-numpy/
		
		series0 = z_norm[vv_ii].copy(); series0 = (series0 - np.nanmean(series0)) / (np.nanstd(series0) + eps)
		series1 = z_norm[adj_list[vv_ii]].copy(); series1 = (series1 - np.nanmean(series1, axis=1)[:,None]) / (np.nanstd(series1, axis=1)[:,None] + eps)
		
		Sxy = np.nanmean(series0[None,:] * series1, axis=1) 
		SxxSxy = np.nanstd(series0)  * np.nanstd(series1, axis=1)
		
		corrs = Sxy / (SxxSxy + 1e-12)
		vertex_means_pearsonr.append(np.nanmean(corrs))
		
	vertex_means_pearsonr = np.hstack(vertex_means_pearsonr)

	return vertex_means_pearsonr

