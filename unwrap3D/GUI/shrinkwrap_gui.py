"""GUI dialog for configuring genus-0 mesh-based shrinkwrap parameters."""

import tkinter as tk
from tkinter import ttk, messagebox


def launch_shrinkwrap_gui(initial_params=None):
    """Launch the shrinkwrap parameter configuration dialog.

    Opens a tkinter window pre-populated with shrinkwrapping parameters.
    Blocks until the user clicks OK or Cancel.

    Parameters
    ----------
    initial_params : dict or None
        Starting parameter values. If None, loads defaults from
        :func:`unwrap3D.Parameters.params.shrinkwrap_genus0_meshbased_params`.

    Returns
    -------
    params : dict or None
        The edited parameter dict if the user clicks OK, or None if cancelled.
    """
    from unwrap3D.Parameters.params import shrinkwrap_genus0_meshbased_params

    if initial_params is None:
        initial_params = shrinkwrap_genus0_meshbased_params()

    # ---- parameter metadata: (key, label, type, widget_type, extra)
    # widget_type: 'entry', 'bool', 'combo'
    # extra: list of choices for combo; ignored otherwise
    PARAM_GROUPS = [
        ("Deformation loop", [
            ('total_shrinkwrap_iters', 'total_shrinkwrap_iters', int,   'entry', None),
            ('decay_rate',            'decay_rate',             float, 'entry', None),
            ('remesh_iters',          'remesh_iters',           int,   'entry', None),
            ('min_size',              'min_size',               int,   'entry', None),
            ('min_lr',                'min_lr',                 float, 'entry', None),
        ]),
        ("Initial alphawrap", [
            ('genus0_alpha_frac',  'genus0_alpha_frac',  float, 'entry', None),
            ('genus0_alpha_auto',  'genus0_alpha_auto',  bool,  'bool',  None),
            ('genus0_tol',         'genus0_tol',         float, 'entry', None),
        ]),
        ("Solver / active contour", [
            ('deltaL',       'deltaL',              float, 'entry', None),
            ('alpha',        'alpha',               float, 'entry', None),
            ('beta',         'beta',                float, 'entry', None),
            ('solver',       'solver',              str,   'combo', ['pardiso', 'scipy']),
            ('smooth_iters', 'smooth_iters (0=off)', int,  'entry', None),
        ]),
        ("Force field", [
            ('anchor_factor',           'anchor_factor',           float, 'entry', None),
            ('concavity_boost_factor',  'concavity_boost_factor',  float, 'entry', None),
            ('force_sigma',             'force_sigma',             float, 'entry', None),
            ('balloon_factor',           'balloon_factor (0=off)',   float, 'entry', None),
        ]),
        ("Convergence / repair", [
            ('convergence_sq_dist',              'convergence_sq_dist',              float, 'entry', None),
            ('topology_repair_alpha_frac',       'topology_repair_alpha_frac',       float, 'entry', None),
            ('n_punchout_refinements',           'n_punchout_refinements',           int,   'entry', None),
            ('vfc_sigma',                        'vfc_sigma (0=off)',                float, 'entry', None),
            ('vfc_blend',                        'vfc_blend',                        float, 'entry', None),
            ('debugviz',                         'debugviz',                         bool,  'bool',  None),
        ]),
    ]

    result = {'value': None}

    root = tk.Tk()
    root.title("Genus-0 Shrinkwrap Parameters")
    root.resizable(False, True)

    # ---- scrollable canvas
    outer = tk.Frame(root)
    outer.pack(fill='both', expand=True)

    canvas = tk.Canvas(outer, width=640, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)

    inner = tk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=inner, anchor='nw')

    def _on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    inner.bind('<Configure>', _on_frame_configure)
    canvas.bind('<Configure>', _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    canvas.bind_all('<MouseWheel>', _on_mousewheel)

    # ---- build parameter widgets
    vars_ = {}   # key -> tk.Variable

    for group_label, params_list in PARAM_GROUPS:
        frame = ttk.LabelFrame(inner, text=group_label, padding=(8, 4))
        frame.pack(fill='x', padx=10, pady=6)

        for key, label, dtype, wtype, extra in params_list:
            row = tk.Frame(frame)
            row.pack(fill='x', pady=2)

            lbl = ttk.Label(row, text=label, width=28, anchor='w')
            lbl.pack(side='left')

            if wtype == 'bool':
                var = tk.BooleanVar(value=bool(initial_params.get(key, False)))
                widget = ttk.Checkbutton(row, variable=var)
                widget.pack(side='left')
            elif wtype == 'combo':
                var = tk.StringVar(value=str(initial_params.get(key, extra[0])))
                widget = ttk.Combobox(row, textvariable=var, values=extra,
                                      state='readonly', width=12)
                widget.pack(side='left')
            else:
                var = tk.StringVar(value=str(initial_params.get(key, '')))
                widget = ttk.Entry(row, textvariable=var, width=14)
                widget.pack(side='left')

            vars_[key] = (var, dtype, wtype)

    # ---- button bar
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill='x', padx=10, pady=8)

    def _reset():
        defaults = shrinkwrap_genus0_meshbased_params()
        for key, (var, dtype, wtype) in vars_.items():
            val = defaults.get(key)
            if wtype == 'bool':
                var.set(bool(val))
            else:
                var.set(str(val))

    def _cancel():
        result['value'] = None
        root.destroy()

    def _ok():
        params = {}
        errors = []
        for key, (var, dtype, wtype) in vars_.items():
            raw = var.get()
            try:
                if wtype == 'bool':
                    params[key] = bool(raw)
                else:
                    params[key] = dtype(raw)
            except (ValueError, TypeError):
                errors.append(f"  {key}: cannot convert '{raw}' to {dtype.__name__}")
        if errors:
            messagebox.showerror("Invalid input",
                                 "Please fix the following fields:\n" + "\n".join(errors),
                                 parent=root)
            return
        result['value'] = params
        root.destroy()

    ttk.Button(btn_frame, text="Reset Defaults", command=_reset).pack(side='left')
    ttk.Button(btn_frame, text="Cancel",         command=_cancel).pack(side='right', padx=(4, 0))
    ttk.Button(btn_frame, text="OK / Accept",    command=_ok).pack(side='right')

    # ---- size and centre
    root.update_idletasks()
    screen_h = root.winfo_screenheight()
    desired_h = min(inner.winfo_reqheight() + btn_frame.winfo_reqheight() + 40, screen_h - 80)
    root.geometry(f"660x{desired_h}")

    root.protocol("WM_DELETE_WINDOW", _cancel)
    root.mainloop()

    return result['value']
