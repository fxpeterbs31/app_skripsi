import streamlit as st
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.markdown('# Halaman Input :writing_hand:')
st.markdown('\n')

img = Image.open('Screenshot 2023-06-05 190129.png')
st.image(img, 
         caption = 'Ilustrasi jembatan gantung tampak depan'
)

# st.markdown('Ini isinya form inputan')
li_inp = []
if 'form_button_clicked' not in st.session_state:
        st.session_state['form_button_clicked'] = False

def callback_form():
    st.session_state.form_button_clicked = True

with st.form('form_1'):
    m = st.number_input(
        label = 'Masukkan Massa Jembatan (kg)',
        min_value = 0,
        value = 2500,
        format = '%d'
    )

    li_inp.append(m)

    l = st.number_input(
        label = 'Masukkan Lebar Jembatan (m)',
        min_value = 0,
        value = 6,
        format = '%d'
    )

    li_inp.append(l)

    k = st.number_input(
        label = 'Masukkan Konstanta Pegas (kg/m)',
        min_value = 0,
        value = 1000,
        format = '%d'
    )

    li_inp.append(k)

    delta = st.number_input(
        label = 'Masukkan Konstanta Peredam Pegas (kg/m)',
        min_value = 0.0,
        value = 0.01,
        step = 0.0001,
        format = '%f'
    )

    li_inp.append(delta)

    alpha = st.number_input(
        label = 'Masukkan Konstanta Nonlinearitas',
        min_value = 0.0,
        value = 0.1,
        step = 0.1,
        format = '%f'
    )

    li_inp.append(alpha)

    h = st.number_input(
        label = 'Masukkan Step Size',
        min_value = 0.0,
        value = 0.01,
        step = 0.01,
        format = '%f'
    )

    li_inp.append(h)

    theta = st.number_input(
        label = 'Masukkan Kemiringan awal jembatan (radian)',
        min_value = 0.0,
        value = 0.01,
        step = 0.01,
        format = '%f'
    )

    li_inp.append(theta)

    v = st.number_input(
        label = 'Masukkan Kecepatan sudut awal jembatan (rad/s)',
        min_value = 0.0,
        value = 0.0,
        step = 0.01,
        format = '%f'
    )

    li_inp.append(v)

    y = st.number_input(
        label = 'Masukkan selisih jarak awal jembatan dan posisi ekuilibrium jembatan (m)',
        min_value = 0.0,
        value = 0.0,
        step = 0.01,
        format = '%f'
    )

    li_inp.append(y)

    w = st.number_input(
        label = 'Masukkan kecepatan awal jembatan (m/s)',
        min_value = 0.0,
        value = 0.0,
        step = 0.01,
        format = '%f'
    )

    li_inp.append(w)

    df_inp = pd.DataFrame(data = {
        'inputs' : li_inp
    })

    if(len(df_inp['inputs'].dropna()) < 6):
        st.write('Masukkan Nilai Pada Semua Kolom!')
    else:
        submit = st.form_submit_button('Masukkan Nilai', on_click = callback_form)

if(submit or st.session_state.form_button_clicked):
    li_params = [
        'Massa Jembatan (m)',
        'Lebar Jembatan (l)',
        'Konstanta Pegas Jembatan (k)', 
        'Konstanta Peredam Pegas Jembatan (\u03B4)', 
        'Konstanta Nonlinearitas (\u03B1)', 
        'Step Size (h)',
        'Kemiringan Jembatan (\u03B8)',
        'Kecepatan Sudut Jembatan (v)',
        'Posisi Jembatan (y)',
        'Kecepatan Jembatan (w)'
    ]

    li_satuan = [
        'kg',
        'm',
        'kg/m',
        'kg/m',
        '',
        '',
        'rad',
        'rad/s',
        'm',
        'm/s'
    ]

    for i in range(10):
        if(i == 0):
            st.markdown('### **PARAMETERS**')
        
        if(i == 6):
            st.write('\n\n')
            st.markdown('### **INITIAL VALUE**')

        col_1, col_2, col_3 = st.columns([0.75,0.15,1])

        with col_1:
            st.write(li_params[i])
        with col_2:
            st.write(': {}'.format(li_inp[i]))
        with col_3:
            st.write(li_satuan[i])

    m     = li_inp[0] 
    l     = li_inp[1] 
    k     = li_inp[2] 
    delta = li_inp[3] 
    alpha = li_inp[4] 
    h     = li_inp[5] 
    theta = li_inp[6]
    v     = li_inp[7]
    y     = li_inp[8]
    w     = li_inp[9]

    if 'inputs' in st.session_state:
        st.session_state['inputs'] = 0

    if 'inputs' not in st.session_state or st.session_state.inputs == 0:
        st.session_state['inputs'] = li_inp

    def f_x_nsfd(x): 
        f = (k/alpha) * (math.exp(alpha*x) - 1)

        return f
        
    def external_force_nsfd(x):
        return 11 * math.sin(3*x)

    def get_alpha_beta_nsfd():
        new_alpha = -alpha/2

        diskriminan_1 = delta*2 - 8*k/m
        diskriminan_2 = delta*2 - 24*k/m

        if(diskriminan_1 < 0 or diskriminan_2 < 0):
            if(diskriminan_1 < 0):
                diskriminan_1 = -diskriminan_1
            if(diskriminan_2 < 0):
                diskriminan_2 = -diskriminan_2
            # st.write(diskriminan_1, diskriminan_2)
            beta_1 = math.sqrt(diskriminan_1)/2
            beta_2 = math.sqrt(diskriminan_2)/2

            status = 'complex'

            return new_alpha, beta_1, beta_2, status
        else:
            eigen_vert_1 = (-alpha + math.sqrt(diskriminan_1))/2
            eigen_vert_2 = (-alpha - math.sqrt(diskriminan_1))/2
            eigen_tors_1 = (-alpha + math.sqrt(diskriminan_2))/2
            eigen_tors_2 = (-alpha - math.sqrt(diskriminan_2))/2

            status = 'not complex'

            return new_alpha, [eigen_vert_1, eigen_vert_2], [eigen_tors_1, eigen_tors_2], status

    def get_phi_nsfd():
        _, _, _, status = get_alpha_beta_nsfd()

        if(status == 'complex'):
            alpha, beta_1, beta_2, status = get_alpha_beta_nsfd()
            numer_1 = (beta_1 * (math.exp(2 * alpha * h) + 1 - 2 * math.exp(alpha * h) * math.cos(beta_1 * h)))
            denom_1 = ((alpha**2 + beta_1**2) * math.exp(alpha*h) * math.sin(beta_1*h))
            phi_1   = numer_1/denom_1

            numer_2 = (beta_2 * (math.exp(2 * alpha * h) + 1 - 2 * math.exp(alpha * h) * math.cos(beta_2 * h)))
            denom_2 = ((alpha**2 + beta_2**2) * math.exp(alpha*h) * math.sin(beta_2*h))
            phi_2   = numer_2/denom_2

            return min(phi_1, phi_2)

        elif(status == 'not complex'):
            alpha, eigen_vert, eigen_tors, status = get_alpha_beta_nsfd()
            eigen_vert_1 = eigen_vert[0]
            eigen_vert_2 = eigen_vert[1]

            eigen_tors_1 = eigen_tors[0]
            eigen_tors_2 = eigen_tors[1]
            
            # eigen_vert
            if(eigen_vert_1 == 0 or eigen_vert_2 == 0):
                phi_vert = h
            elif(eigen_vert_1 == eigen_vert_2 and eigen_vert_1 != 0 and eigen_vert_2 != 0):
                phi_vert = (math.exp(eigen_vert_1*h) - 1)**2/(eigen_vert_1**2 * h * math.exp(eigen_vert_1*h))
            elif(eigen_vert_1 != eigen_vert_2 and eigen_vert_1*eigen_vert_2 != 0):
                phi_vert = (eigen_vert_1 - eigen_vert_2)*(math.exp(eigen_vert_1*h)-1)*(math.exp(eigen_vert_2*h)-1)/(eigen_vert_1*eigen_vert_2)*(math.exp(eigen_vert_1*h) - math.exp(eigen_vert_2*h))
            # eigen tors
            if(eigen_tors_1 == 0 or eigen_tors_2 == 0):
                phi_tors = h
            elif(eigen_tors_1 == eigen_tors_2 and eigen_tors_1 != 0 and eigen_tors_2 != 0):
                phi_tors = (math.exp(eigen_tors_1*h) - 1)**2/(eigen_tors_1**2 * h * math.exp(eigen_tors_1*h))
            elif(eigen_tors_1 != eigen_tors_2 and eigen_tors_1*eigen_tors_2 != 0):
                phi_tors = (eigen_tors_1 - eigen_tors_2)*(math.exp(eigen_tors_1*h)-1)*(math.exp(eigen_tors_2*h)-1)/(eigen_tors_1*eigen_tors_2)*(math.exp(eigen_tors_1*h) - math.exp(eigen_tors_2*h))
            
            return min(phi_vert, phi_tors)

    def get_numeric_results_nsfd(u, ivp):
        m, l, k, delta = u
        theta_0, v_0, y_0, w_0 = ivp
        time_units = 150
        
        li_y     = []
        li_w     = []
        li_theta = []
        li_v     = []
        li_h     = []

        loops = int(time_units / h)
        for i in range(loops):
            phi    = get_phi_nsfd()

            y     = (phi * w_0) + y_0
            theta = (phi * v_0) + theta_0

            min_delta = delta * -1

            minus = y - l * math.sin(theta)
            plus  = y + l * math.sin(theta)

            f_min  = f_x_nsfd(minus) - f_x_nsfd(plus)
            f_plus = f_x_nsfd(minus) + f_x_nsfd(plus)

            ext    = external_force_nsfd(h*i)
            cosine = math.cos(theta)

            w     = (phi * (min_delta * w_0 - f_plus/m + ext)) + w_0
            v     = (phi * (min_delta * v_0 + (3/(m*l)) * cosine * f_min)) + v_0

            li_y.append(y)
            li_w.append(w)
            li_theta.append(theta)
            li_v.append(v)
            li_h.append(h*i)

            y_0     = y
            w_0     = w
            theta_0 = theta
            v_0     = v
            
        return li_y, li_theta, li_h

    # Euler
    class ForwardEuler:
        def __init__(self, f):
            self.f = lambda t, u: np.asarray(f(t, u), float)
            
        def set_initial_condition(self, u0):
            if isinstance(u0, (float, int)):
                self.neq = 1 #number of eqs.
                u0 = float(u0)
            else:
                u0 = np.asarray(u0)
                self.neq = u0.size
                
            self.u0 = u0
            
        def solve(self, t_span, N):
            """
            Compute solution for
            t_span[0] <= t <= t_span[1],
            using N steps.
            """
            t0, T = t_span
            self.dt = (T - t0)/N
            self.t = np.zeros(N+1)
            
            if self.neq == 1:
                self.u = np.zeros(N+1)
            else:
                self.u = np.zeros((N+1, self.neq))
                
            self.t[0] = t0
            self.u[0] = self.u0
            
            for n in range(N):
                self.n = n
                self.t[n+1] = self.t[n] + self.dt
                self.u[n+1] = self.advance()
            
            return self.t, self.u
        def advance(self):
            """Advance the solution one time step."""
            u, dt, f, n, t = self.u, self.dt, self.f, self.n, self.t
            unew = u[n] + dt*f(t[n], u[n])
            
            return unew
        
    class Tacoma:
        def __init__(self, k, l, m, h, delta, alpha, theta, v, y, w):
            self.k     = k
            self.l     = l 
            self.m     = m 
            self.h     = h
            self.delta = delta 
            self.alpha = alpha
            self.theta = theta
            self.v     = v
            self.y     = y
            self.w     = w
            
        def f_x(self, x): 
            f = (self.k/self.alpha) * (math.exp(self.alpha*x) - 1)

            return f

        def external_force(self, x):
            return 11 * math.sin(3*x)
            
        def __call__(self, t, u):
            y, w, theta, v = u
            
            min_delta = self.delta * -1
        
            minus = y - self.l * math.sin(theta)
            plus  = y + self.l * math.sin(theta)

            f_min  = self.f_x(minus) - self.f_x(plus)
            f_plus = self.f_x(minus) + self.f_x(plus)

            ext    = self.external_force(t)
            cosine = math.cos(theta)
            
            dy     = w
            dw     = (min_delta * w) - (f_plus / self.m) + (ext)
            
            dtheta = v
            dv     = (min_delta * v) + ((3/(self.m*self.l)) * cosine * f_min)
            
            return [dy, dw, dtheta, dv]

    li_y, li_theta, li_h = get_numeric_results_nsfd(
        u = (m, l, k, delta),
        ivp = (theta, v, y, w)
    )

    ## Torsional NSFD
    df_theta_nsfd = pd.DataFrame(data = {
        't' : li_h,
        'theta' : li_theta
    }).set_index('t')

    df_theta_nsfd_to_display = df_theta_nsfd.reset_index()
    df_theta_nsfd_to_display = df_theta_nsfd_to_display.astype('string')
    df_theta_nsfd_to_display['t']     = df_theta_nsfd_to_display['t']     + ' s'
    df_theta_nsfd_to_display['theta'] = df_theta_nsfd_to_display['theta'] + ' radian'
    df_theta_nsfd_to_display = df_theta_nsfd_to_display.set_index('t')

    ## Vertikal NSFD
    df_y_nsfd = pd.DataFrame(data = {
        't' : li_h,
        'y' : li_y
    }).set_index('t')

    df_y_nsfd_to_display = df_y_nsfd.reset_index()
    df_y_nsfd_to_display = df_y_nsfd_to_display.astype('string')
    df_y_nsfd_to_display['t'] = df_y_nsfd_to_display['t'] + ' s'
    df_y_nsfd_to_display['y'] = df_y_nsfd_to_display['y'] + ' m'
    df_y_nsfd_to_display = df_y_nsfd_to_display.set_index('t')

    # Euler
    problem = Tacoma(
        k     = k,
        l     = l,
        m     = m,
        h     = h,
        delta = delta,
        alpha = alpha,
        theta = theta,
        v     = v,
        y     = y,
        w     = w
    )

    solver = ForwardEuler(problem)
    # y, w, theta, v = 0, 0, 0.01, 0
    solver.set_initial_condition((
        y, w, theta, v
    ))
    T = 150
    N = int(T/h)
    t, u = solver.solve(t_span = (0, T), N = N)

    ## Torsional Euler
    df_theta_euler = pd.DataFrame(data = {
        't' : t,
        'theta' : u[:, 2]
    }).set_index('t')

    df_theta_euler_to_display = df_theta_euler.reset_index()
    df_theta_euler_to_display = df_theta_euler_to_display.astype('string')
    df_theta_euler_to_display['t']     = df_theta_euler_to_display['t']     + ' s'
    df_theta_euler_to_display['theta'] = df_theta_euler_to_display['theta'] + ' radian'
    df_theta_euler_to_display = df_theta_euler_to_display.set_index('t')

    ## Vertikal Euler
    df_y_euler = pd.DataFrame(data = {
        't' : t,
        'y' : u[:, 0]
    }).set_index('t')

    df_y_euler_to_display = df_y_euler.reset_index()
    df_y_euler_to_display = df_y_euler_to_display.astype('string')
    df_y_euler_to_display['t'] = df_y_euler_to_display['t'] + ' s'
    df_y_euler_to_display['y'] = df_y_euler_to_display['y'] + ' m'
    df_y_euler_to_display = df_y_euler_to_display.set_index('t')

    if 'button_clicked' not in st.session_state:
        st.session_state['button_clicked'] = False

    def callback():
        st.session_state.button_clicked = True

    if(st.button('Tampilkan Hasil Komputasi', on_click = callback) or st.session_state.button_clicked):
        choose_motion = st.radio(
            'Pilih Gerakan yang Ingin Ditampilkan',
            ['Torsional', 'Vertikal'],
            horizontal = True
        )

        choose_method = st.radio(
            'Pilih Metode yang Ingin Digunakan',
            ['NSFD', 'Euler'],
            horizontal = True
        )

        li_display = []
        li_params = ['Grafik', 'Tabel']

        tab_1, tab_2 = st.tabs(li_params)
        
        if(choose_motion == 'Torsional' and choose_method == 'NSFD'):   
            with tab_1:
                st.line_chart(df_theta_nsfd)
                name_plot = 'Torsional_NSFD_Plot.png'

                plt.plot(df_theta_nsfd)
                plt.title('Torsional NSFD Plot')
                plt.savefig(name_plot)

                with open(name_plot, 'rb') as img:
                        btn = st.download_button(
                                label="Download Plot as PNG",
                                data=img,
                                file_name=name_plot,
                                mime="image/png"
                        )

            with tab_2:
                _, table, _ = st.columns([2.5, 5, 2.5])
                with table:
                    st.dataframe(df_theta_nsfd_to_display, width = 280)

                name_table = 'Torsional NSFD Excel.csv'
                btn_csv = st.download_button(
                    label = 'Download Table as CSV / Excel',                    
                    data = df_theta_nsfd.to_csv(),
                    file_name = name_table,
                    mime = 'text/csv'
                )

        elif(choose_motion == 'Torsional' and choose_method == 'Euler'):
            with tab_1:
                st.line_chart(df_theta_euler)
                name_plot = 'Torsional_Euler_Plot.png'

                plt.plot(df_theta_euler)
                plt.title('Torsional Euler Plot')
                plt.savefig(name_plot)

                with open(name_plot, 'rb') as img:
                        btn = st.download_button(
                                label="Download Plot as PNG",
                                data=img,
                                file_name=name_plot,
                                mime="image/png"
                        )

            with tab_2:
                _, table, _ = st.columns([2.5, 5, 2.5])
                with table:
                    st.dataframe(df_theta_euler_to_display, width = 280)

                name_table = 'Torsional Euler Excel.csv'
                btn_csv = st.download_button(
                    label = 'Download Table as CSV / Excel',                    
                    data = df_theta_euler.to_csv(),
                    file_name = name_table,
                    mime = 'text/csv'
                )

        elif(choose_motion == 'Vertikal' and choose_method == 'NSFD'):
            with tab_1:
                st.line_chart(df_y_nsfd)
                name_plot = 'y_NSFD_Plot.png'

                plt.plot(df_y_nsfd)
                plt.title('Vertical NSFD Plot')
                plt.savefig(name_plot)

                with open(name_plot, 'rb') as img:
                        btn = st.download_button(
                                label="Download Plot as PNG",
                                data=img,
                                file_name=name_plot,
                                mime="image/png"
                        )

            with tab_2:
                _, table, _ = st.columns([2.5, 5, 2.5])
                with table:
                    st.dataframe(df_y_nsfd_to_display, width = 280)

                name_table = 'Vertical NSFD Excel.csv'
                btn_csv = st.download_button(
                    label = 'Download Table as CSV / Excel',                    
                    data = df_y_nsfd.to_csv(),
                    file_name = name_table,
                    mime = 'text/csv'
                )

        elif(choose_motion == 'Vertikal' and choose_method == 'Euler'):
            with tab_1:
                st.line_chart(df_y_euler)
                name_plot = 'y_Euler_Plot.png'

                plt.plot(df_y_euler)
                plt.title('Vertical Euler Plot')
                plt.savefig(name_plot)

                with open(name_plot, 'rb') as img:
                        btn = st.download_button(
                                label="Download Plot as PNG",
                                data=img,
                                file_name=name_plot,
                                mime="image/png"
                        )

            with tab_2:
                _, table, _ = st.columns([2.5, 5, 2.5])
                with table:
                    st.dataframe(df_y_euler_to_display, width = 280)

                name_table = 'Vertical Euler Excel.csv'
                btn_csv = st.download_button(
                    label = 'Download Table as CSV / Excel',                    
                    data = df_y_euler.to_csv(),
                    file_name = name_table,
                    mime = 'text/csv'
                )

