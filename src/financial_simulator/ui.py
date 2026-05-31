import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import json

dist_types = {
    'NormalDistribution': ['mean', 'std'],
    'UniformDistribution': ['low', 'high'],
    'TriangularDistribution': ['low', 'mode', 'high'],
    'DateDistribution': ['start', 'end'],
}

timing_types = {
    'OneTime': ['time'],
    'Interval': ['interval_days', 'start_time'],
    'Random': ['start', 'end', 'n'],
    'Seasonal': ['months'],  # plus inner
}

value_gen_types = {
    'Fixed': ['value'],
    'Growing': ['initial', 'growth_rate'],
    'Distribution': [],  # plus dist
    'RateChange': ['update_key'],  # plus dist
    'VariableRateLoan': ['principal', 'initial_rate', 'term_months', 'rate_key'],
}

continuous_types = {
    'Appreciation': ['rate', 'var'],
}

def parse_value(val_str):
    try:
        return float(val_str)
    except ValueError:
        try:
            return int(val_str)
        except ValueError:
            return val_str

def create_dist_form(parent):
    frame = ttk.Frame(parent)
    tk.Label(frame, text="Type:").pack(side=tk.LEFT)
    type_combo = ttk.Combobox(frame, values=list(dist_types.keys()))
    type_combo.pack(side=tk.LEFT)
    param_frame = ttk.Frame(frame)
    param_frame.pack(fill=tk.X)

    def on_change(e):
        for child in param_frame.winfo_children():
            child.destroy()
        typ = type_combo.get()
        if typ:
            params = dist_types[typ]
            row = 0
            for p in params:
                tk.Label(param_frame, text=p + ":").grid(row=row, column=0, sticky='w')
                entry = tk.Entry(param_frame)
                entry.grid(row=row, column=1, sticky='w')
                row += 1

    type_combo.bind("<<ComboboxSelected>>", on_change)

    def get_dict():
        typ = type_combo.get()
        if not typ:
            return None
        d = {'type': typ}
        children = param_frame.winfo_children()
        for i in range(0, len(children), 2):
            label = children[i]
            entry = children[i+1]
            key = label['text'][:-1]
            d[key] = parse_value(entry.get())
        return d

    frame.pack(fill=tk.X)
    return frame, get_dict

def create_timing_form(parent):
    frame = ttk.Frame(parent)
    tk.Label(frame, text="Timing Type:").pack(anchor='w')
    type_combo = ttk.Combobox(frame, values=list(timing_types.keys()))
    type_combo.pack(anchor='w')
    param_frame = ttk.Frame(frame)
    param_frame.pack(fill=tk.X)
    inner_frame = None
    inner_get = None

    def on_change(e):
        nonlocal inner_frame, inner_get
        for child in param_frame.winfo_children():
            child.destroy()
        if inner_frame:
            inner_frame.destroy()
            inner_frame = None
            inner_get = None
        typ = type_combo.get()
        if typ:
            params = timing_types[typ]
            row = 0
            for p in params:
                tk.Label(param_frame, text=p + ":").grid(row=row, column=0, sticky='w')
                entry = tk.Entry(param_frame)
                entry.grid(row=row, column=1, sticky='w')
                row += 1
            if typ == 'Seasonal':
                inner_frame = ttk.Frame(frame)
                inner_frame.pack(fill=tk.X)
                tk.Label(inner_frame, text="Inner Timing").pack(anchor='w')
                _, inner_get = create_timing_form(inner_frame)

    type_combo.bind("<<ComboboxSelected>>", on_change)

    def get_dict():
        typ = type_combo.get()
        if not typ:
            return None
        d = {'type': typ}
        children = param_frame.winfo_children()
        for i in range(0, len(children), 2):
            label = children[i]
            entry = children[i+1]
            key = label['text'][:-1]
            val_str = entry.get()
            val = parse_value(val_str)
            if key == 'months':
                val = [int(m.strip()) for m in val_str.split(',') if m.strip()]
            elif key in ['interval_days', 'n']:
                val = int(val_str) if val_str else None
            d[key] = val
        if typ == 'Seasonal' and inner_get:
            d['inner'] = inner_get()
        return d

    frame.pack(fill=tk.X)
    return frame, get_dict

def create_value_gen_form(parent):
    frame = ttk.Frame(parent)
    tk.Label(frame, text="Value Gen Type:").pack(anchor='w')
    type_combo = ttk.Combobox(frame, values=list(value_gen_types.keys()))
    type_combo.pack(anchor='w')
    param_frame = ttk.Frame(frame)
    param_frame.pack(fill=tk.X)
    dist_frame = None
    get_dist = None

    def on_change(e):
        nonlocal dist_frame, get_dist
        for child in param_frame.winfo_children():
            child.destroy()
        if dist_frame:
            dist_frame.destroy()
            dist_frame = None
            get_dist = None
        typ = type_combo.get()
        if typ:
            params = value_gen_types[typ]
            row = 0
            for p in params:
                tk.Label(param_frame, text=p + ":").grid(row=row, column=0, sticky='w')
                entry = tk.Entry(param_frame)
                entry.grid(row=row, column=1, sticky='w')
                row += 1
            if typ in ['Distribution', 'RateChange']:
                dist_frame = ttk.Frame(frame)
                dist_frame.pack(fill=tk.X)
                tk.Label(dist_frame, text="Distribution").pack(anchor='w')
                _, get_dist = create_dist_form(dist_frame)

    type_combo.bind("<<ComboboxSelected>>", on_change)

    def get_dict():
        typ = type_combo.get()
        if not typ:
            return None
        d = {'type': typ}
        children = param_frame.winfo_children()
        for i in range(0, len(children), 2):
            label = children[i]
            entry = children[i+1]
            key = label['text'][:-1]
            d[key] = parse_value(entry.get())
        if get_dist:
            d['dist'] = get_dist()
        return d

    frame.pack(fill=tk.X)
    return frame, get_dict

def create_key_value_form(parent, title="Add Key-Value Pairs"):
    frame = ttk.Frame(parent)
    tk.Label(frame, text=title).pack(anchor='w')
    pairs = []

    def add_pair():
        subframe = ttk.Frame(frame)
        subframe.pack(fill=tk.X)
        key_entry = tk.Entry(subframe)
        key_entry.pack(side=tk.LEFT, padx=5)
        val_entry = tk.Entry(subframe)
        val_entry.pack(side=tk.LEFT, padx=5)
        remove_btn = ttk.Button(subframe, text="Remove", command=lambda: (subframe.destroy(), pairs.remove((key_entry, val_entry))))
        remove_btn.pack(side=tk.LEFT)
        pairs.append((key_entry, val_entry))

    add_btn = ttk.Button(frame, text="Add Pair", command=add_pair)
    add_btn.pack(anchor='w')

    def get_dict():
        return {k.get(): parse_value(v.get()) for k, v in pairs if k.get()}

    frame.pack(fill=tk.X)
    return frame, get_dict

class ConfigGeneratorUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Financial Simulation Config Generator")
        self.geometry("1000x700")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        self.dists_getters = []
        self.cont_getters = []
        self.builders_getters = []

        self.create_general_tab()
        self.create_dists_tab()
        self.create_initial_state_tab()
        self.create_continuous_tab()
        self.create_builders_tab()

        save_btn = ttk.Button(self, text="Save JSON", command=self.save_config)
        save_btn.pack(pady=10)

    def create_general_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="General")
        row = 0
        tk.Label(tab, text="Num Simulations:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.num_sims = tk.StringVar(value="1000")
        tk.Entry(tab, textvariable=self.num_sims).grid(row=row, column=1, sticky='w')
        row += 1
        tk.Label(tab, text="Seed:").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.seed = tk.StringVar(value="42")
        tk.Entry(tab, textvariable=self.seed).grid(row=row, column=1, sticky='w')
        row += 1
        tk.Label(tab, text="Simulation Start (ISO):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.sim_start = tk.StringVar(value="2026-01-01T00:00:00")
        tk.Entry(tab, textvariable=self.sim_start).grid(row=row, column=1, sticky='w')
        row += 1
        tk.Label(tab, text="Simulation End (ISO):").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.sim_end = tk.StringVar(value="2036-01-01T00:00:00")
        tk.Entry(tab, textvariable=self.sim_end).grid(row=row, column=1, sticky='w')

    def create_dists_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Distributions")
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        self.dists_inner = ttk.Frame(canvas)
        self.dists_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.dists_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        add_btn = ttk.Button(tab, text="Add Distribution", command=self.add_dist)
        add_btn.pack(pady=5)

    def add_dist(self):
        frame = ttk.Frame(self.dists_inner, relief='raised', borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(frame, text="Name:").grid(row=0, column=0, sticky='w')
        name_entry = tk.Entry(frame)
        name_entry.grid(row=0, column=1, sticky='w')
        tk.Label(frame, text="Type:").grid(row=0, column=2, sticky='w')
        type_combo = ttk.Combobox(frame, values=list(dist_types.keys()))
        type_combo.grid(row=0, column=3, sticky='w')
        param_frame = ttk.Frame(frame)
        param_frame.grid(row=1, column=0, columnspan=4, sticky='w')

        def on_change(e):
            for child in param_frame.winfo_children():
                child.destroy()
            typ = type_combo.get()
            if typ:
                params = dist_types[typ]
                col = 0
                for p in params:
                    tk.Label(param_frame, text=p + ":").grid(row=0, column=col, sticky='w')
                    entry = tk.Entry(param_frame)
                    entry.grid(row=0, column=col+1, sticky='w')
                    col += 2

        type_combo.bind("<<ComboboxSelected>>", on_change)

        def get_this():
            name = name_entry.get()
            if not name:
                return None
            typ = type_combo.get()
            d = {'type': typ}
            children = param_frame.winfo_children()
            for i in range(0, len(children), 2):
                label = children[i]
                entry = children[i+1]
                key = label['text'][:-1]
                d[key] = parse_value(entry.get())
            return name, d

        def remove():
            frame.destroy()
            self.dists_getters.remove(get_this)

        remove_btn = ttk.Button(frame, text="Remove", command=remove)
        remove_btn.grid(row=0, column=4, sticky='w')
        self.dists_getters.append(get_this)

    def create_initial_state_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Initial State")
        _, self.initial_state_get = create_key_value_form(tab, title="Initial State Key-Values")

    def create_continuous_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Continuous Processes")
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        self.cont_inner = ttk.Frame(canvas)
        self.cont_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.cont_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        add_btn = ttk.Button(tab, text="Add Process", command=self.add_cont)
        add_btn.pack(pady=5)

    def add_cont(self):
        frame = ttk.Frame(self.cont_inner, relief='raised', borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(frame, text="Type:").grid(row=0, column=0, sticky='w')
        type_combo = ttk.Combobox(frame, values=list(continuous_types.keys()))
        type_combo.grid(row=0, column=1, sticky='w')
        param_frame = ttk.Frame(frame)
        param_frame.grid(row=1, column=0, columnspan=2, sticky='w')

        def on_change(e):
            for child in param_frame.winfo_children():
                child.destroy()
            typ = type_combo.get()
            if typ:
                params = continuous_types[typ]
                row = 0
                for p in params:
                    tk.Label(param_frame, text=p + ":").grid(row=row, column=0, sticky='w')
                    entry = tk.Entry(param_frame)
                    entry.grid(row=row, column=1, sticky='w')
                    row += 1

        type_combo.bind("<<ComboboxSelected>>", on_change)

        def get_this():
            typ = type_combo.get()
            if not typ:
                return None
            d = {'type': typ}
            children = param_frame.winfo_children()
            for i in range(0, len(children), 2):
                label = children[i]
                entry = children[i+1]
                key = label['text'][:-1]
                d[key] = parse_value(entry.get())
            return d

        def remove():
            frame.destroy()
            self.cont_getters.remove(get_this)

        remove_btn = ttk.Button(frame, text="Remove", command=remove)
        remove_btn.grid(row=0, column=2, sticky='w')
        self.cont_getters.append(get_this)

    def create_builders_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Builders")
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        self.builders_inner = ttk.Frame(canvas)
        self.builders_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.builders_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        add_btn = ttk.Button(tab, text="Add Builder", command=self.add_builder)
        add_btn.pack(pady=5)

    def add_builder(self):
        frame = ttk.Frame(self.builders_inner, relief='raised', borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=5)
        tk.Label(frame, text="Name:").grid(row=0, column=0, sticky='w')
        name_entry = tk.Entry(frame)
        name_entry.grid(row=0, column=1, sticky='w')

        timing_frame = ttk.Frame(frame)
        timing_frame.grid(row=1, column=0, columnspan=3, sticky='w')
        tk.Label(timing_frame, text="Timing").pack(anchor='w')
        _, get_timing = create_timing_form(timing_frame)

        value_frame = ttk.Frame(frame)
        value_frame.grid(row=2, column=0, columnspan=3, sticky='w')
        tk.Label(value_frame, text="Value Generator").pack(anchor='w')
        _, get_value = create_value_gen_form(value_frame)

        meta_frame = ttk.Frame(frame)
        meta_frame.grid(row=3, column=0, columnspan=3, sticky='w')
        _, get_meta = create_key_value_form(meta_frame, title="Metadata")

        def get_this():
            name = name_entry.get()
            if not name:
                return None
            return {
                'name': name,
                'timing': get_timing(),
                'value_gen': get_value(),
                'metadata': get_meta()
            }

        def remove():
            frame.destroy()
            self.builders_getters.remove(get_this)

        remove_btn = ttk.Button(frame, text="Remove", command=remove)
        remove_btn.grid(row=0, column=2, sticky='w')
        self.builders_getters.append(get_this)

    def save_config(self):
        config = {
            'num_simulations': int(self.num_sims.get()),
            'seed': int(self.seed.get()),
            'dists': {},
            'simulation': {
                'start': self.sim_start.get(),
                'end': self.sim_end.get(),
                'initial_state': self.initial_state_get(),
                'continuous_processes': [get() for get in self.cont_getters if get()],
                'builders': [get() for get in self.builders_getters if get()],
            }
        }
        for get in self.dists_getters:
            res = get()
            if res:
                name, d = res
                config['dists'][name] = d

        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)

if __name__ == "__main__":
    app = ConfigGeneratorUI()
    app.mainloop()