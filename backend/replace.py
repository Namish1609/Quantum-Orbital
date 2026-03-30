import sys, re

with open('main.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = '''
    try:
        # PURE SPHERICAL REJECTION SAMPLING
        r_c = np.linspace(0.1, size, 100)
        theta_c = np.pi/2
        phi_c = 0.0
        psi_c = hydrogen_wavefunction(r_c, theta_c, phi_c, n, l, m, Z)
        max_p = np.max(np.abs(psi_c)**2) * 1.5
        if max_p == 0: max_p = 1.0

        accepted_x, accepted_y, accepted_z, accepted_d, accepted_phase = [], [], [], [], []
        batch_size = 500000
        max_iterations = 40
        
        for _ in range(max_iterations):
            if len(accepted_x) >= num_points: break
            
            u_r = np.random.rand(batch_size)
            r_b = size * np.cbrt(u_r)
            
            u_theta = np.random.rand(batch_size)
            theta_b = np.arccos(2 * u_theta - 1)
            
            u_phi = np.random.rand(batch_size)
            phi_b = 2 * np.pi * u_phi

            psi_b = hydrogen_wavefunction(r_b, theta_b, phi_b, n, l, m, Z)
            d_b = np.abs(psi_b)**2
            p_b = np.power(d_b, density_scale)
            
            u = np.random.uniform(0, max_p, batch_size)
            mask = u < p_b
            
            x_b = r_b[mask] * np.sin(theta_b[mask]) * np.cos(phi_b[mask])
            y_b = r_b[mask] * np.sin(theta_b[mask]) * np.sin(phi_b[mask])
            z_b = r_b[mask] * np.cos(theta_b[mask])
            
            accepted_x.extend(x_b)
            accepted_y.extend(y_b)
            accepted_z.extend(z_b)
            accepted_d.extend(d_b[mask])
            
            phase = np.sign(np.real(psi_b[mask]))
            accepted_phase.extend(phase)

        accepted_x = np.array(accepted_x[:num_points])
        accepted_y = np.array(accepted_y[:num_points])
        accepted_z = np.array(accepted_z[:num_points])
        accepted_d = np.array(accepted_d[:num_points])
        accepted_phase = np.array(accepted_phase[:num_points])
        
        if len(accepted_d) > 0:
            d_max = np.max(accepted_d)
            if d_max > 0: accepted_d = accepted_d / d_max
                
        points = np.column_stack((accepted_x, accepted_y, accepted_z, accepted_d, accepted_phase))
        return {'points': points.tolist()}

    except Exception as e:
'''
text = re.sub(r'    try:.*?except Exception as e:', replacement, text, flags=re.DOTALL)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(text)
