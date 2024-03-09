#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:28:39 2023

@author: mnv
"""

import fenics as fen
from ufl.operators import diff
from ufl import variable
import numpy as np
import sympy as sp

comm = fen.MPI.comm_world

rank = comm.Get_rank()
size = comm.Get_size()

def print_0(*obj):
    if rank == 0:
        print(*obj)

class llg2_solver:
    """Solver for 2d LLG problem"""
    
    def __init__(self, Lx, Ly, mesh):
        """Initialize with mesh, construct Function Space, trial function, test function"""
        
        self.Lx = Lx
        self.Ly = Ly
        self.mesh = mesh
        
        Elv = fen.VectorElement('CG', fen.triangle, 1, dim = 3)
        self.FS = fen.FunctionSpace(mesh, Elv)
        
        Elf = fen.FiniteElement("CG", fen.triangle, degree = 1)
        self.FS_1d = fen.FunctionSpace(mesh, Elf)
        
        self.v = fen.Function(self.FS) #TrialFunction
        self.w = fen.TestFunction(self.FS)
        
        self.vp = fen.TrialFunction(self.FS_1d)
        self.wp = fen.TestFunction(self.FS_1d)
        
        self.pot = fen.Function(self.FS_1d)
        
        self.h_ext = None
        
    # def norm_sol_s(self, u):
    #     #vector().array() is replaced by vector().get_local()
    #     u_array = u.vector().get_local()
    #     N = int(np.size(u_array))
    #     u_array_2 = np.zeros(u_array.shape)
    #     i = 0
    #     while i+2 < N:
    #         norm = np.sqrt(u_array[i]**2 + u_array[i+1]**2 + u_array[i+2]**2)
    #         u_array_2[i] = u_array[i] / norm
    #         u_array_2[i+1] = u_array[i+1] / norm
    #         u_array_2[i+2] = u_array[i+2] / norm
    #         i += 3
            
    #     u.vector()[:] = u_array_2
    #     return u
    
    def norm_sol_s(self, u):
        #vector().array() is replaced by vector().get_local()
        u_array = u.vector().get_local()
        N = np.size(u_array)//3
        u_array = np.reshape(u_array, (N, 3))
        
        norm = np.linalg.norm(u_array, axis = 1)
        
        u_array = u_array/np.expand_dims(norm, axis = 1)
        
        u.vector()[:] = u_array.reshape((N*3,))
        return u
    
    def demag_pot(self):
        #self.pot = fen.Function(self.FS_1d)
        self.pot.vector()[:] = self.phi_prev.vector()
        U = self.pot.vector()
    
        self.solver_p.solve(self.A_p, U, self.b_p)
        self.pot.vector()[:] = U
    
    def max_norm(self, u):
        u1, u2, u3 = u.split()
        V1 = u1.compute_vertex_values()
        V2 = u2.compute_vertex_values()
        V3 = u3.compute_vertex_values()
        norm_prev = np.max(np.sqrt(V1*V1 + V2*V2 + V3*V3))
        return norm_prev
        
    def set_params(self, alpha = 1, kku = 1000, A_ex = 9.5*10**(-8), Ms = 4, pin = True):
        """Set parameters"""
        
        self.alpha = alpha
        self.kku = kku
        self.Ms = Ms
        self.A_ex = A_ex
        self.dw_width = np.sqrt(A_ex/kku)
        
        x_a = 5
        delta_x = 3
        tol = 1E-14
        
        class Omega_0(fen.SubDomain):
            def inside(self, x, on_boundary):
                return (np.abs(x[0] - x_a) <= delta_x/2 + tol)
            
        materials = fen.MeshFunction('size_t', self.mesh, dim = 1)
        subdomain_0 = Omega_0()
        subdomain_0.mark(materials, 1)
        
        class KuFunc(fen.UserExpression):
            def __init__(self, materials, ku_0, ku_1, **kwargs):
                super().__init__(**kwargs)
                self.materials = materials
                self.ku_0 = ku_0
                self.ku_1 = ku_1
        
            def eval_cell(self, values, x, cell):
                if self.materials[cell.index] == 0:
                    values[0] = self.ku_0
                else:
                    values[0] = self.ku_1
        
        ku_func_expr = KuFunc(materials, 1, 0.85, degree = 0)
        
        El_DP = fen.FiniteElement('DP', fen.triangle, 0)
        FS_DP = fen.FunctionSpace(self.mesh, El_DP)
        
        if pin == True:
            self.ku_func = fen.project(ku_func_expr, FS_DP)
        else:
            self.ku_func = fen.project(fen.Expression('1.', degree = 2), FS_DP)
    
    def e_field_from_ps(self, xx0 = 0, yy0 = 0, rr0 = 0.00002, UU0 = 2*10/3/50, gamma_me = 1E-6):
        """
        Construct electric field from tip with radius rr0 and potential UU0

        Parameters
        ----------
        xx0 : TYPE, optional
            DESCRIPTION. The default is 0.
        yy0 : TYPE, optional
            DESCRIPTION. The default is 0.
        rr0 : TYPE, optional
            DESCRIPTION. The default is 0.00002.
        UU0 : TYPE, optional
            DESCRIPTION. The default is 10/3/50.
        gamma_me : TYPE, optional
            DESCRIPTION. The default is 1E-6.

        Returns
        -------
        None.

        """
        
        self.p = gamma_me*UU0/rr0/(2*np.sqrt(self.A_ex*self.kku))
        
        x, y, z = sp.symbols('x y z')
        xx, yy = sp.symbols('x[0] x[1]')
        x0, y0 = sp.symbols('x0 y0')
        d, r0, U0 = sp.symbols('d r0 U0')
        f_expr = U0*r0/sp.sqrt((r0-z)**2+((x-x0)**2 + (y-y0)**2))
        
        E1 = -sp.diff(f_expr,x)
        E1 = (E1.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(x,xx),(y,yy)])/U0*r0)
        dE1_dz = sp.diff(E1,z)
        dE1_dy = sp.diff(E1,y)
        E1 = E1.subs([(z,0)])
        dE1_dz = dE1_dz.subs([(z,0)])
        dE1_dy = dE1_dy.subs([(z,0)])

        E2 = -sp.diff(f_expr,y)
        E2 = (E2.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(x,xx),(y,yy)])/U0*r0)
        dE2_dz = sp.diff(E2,z)
        dE2_dy = sp.diff(E2,y)
        E2 = E2.subs([(z,0)])
        dE2_dz = dE2_dz.subs([(z,0)])
        dE2_dy = dE2_dy.subs([(z,0)])

        E3 = -sp.diff(f_expr,z)
        E3 = (E3.subs([(x,d*x),(y,d*y),(z,d*z),(x0,d*x0),(y0,d*y0),(x,xx),(y,yy)])/U0*r0)
        dE3_dz = sp.diff(E3,z)
        dE3_dy = sp.diff(E3,y)
        E3 = E3.subs([(z,0)])
        dE3_dz = dE3_dz.subs([(z,0)])
        dE3_dy = dE3_dy.subs([(z,0)])

        E1_c=sp.ccode(E1)
        E2_c=sp.ccode(E2)
        E3_c=sp.ccode(E3)

        dE1_dz_c = sp.ccode(dE1_dz)
        dE2_dz_c = sp.ccode(dE2_dz)
        dE3_dz_c = sp.ccode(dE3_dz)
        
        dE1_dy_c = sp.ccode(dE1_dy)
        dE2_dy_c = sp.ccode(dE2_dy)
        dE3_dy_c = sp.ccode(dE3_dy)
        
        
        # e1 = fen.Expression((E1_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)   
        # e2 = fen.Expression((E2_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)
        # e3 = fen.Expression((E3_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)
        e_v = fen.Expression((E1_c, E2_c, E3_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)
        self.e_v = fen.project(e_v, self.FS)
        
        dedz_v = fen.Expression((dE1_dz_c, dE2_dz_c, dE3_dz_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)
        self.dedz_v = fen.project(dedz_v, self.FS)
        
        dedy_v = fen.Expression((dE1_dy_c, dE2_dy_c, dE3_dy_c), degree = 2, U0 = UU0, d = self.dw_width, r0 = rr0, x0 = xx0, y0 = yy0)
        self.dedy_v = fen.project(dedy_v, self.FS)
        
        print("ME parameter p = ", self.p)
    
    def set_h_ext(self, h_x = -13, h_y = 0, h_z = 0):
        """
        Set external magnetic field

        Parameters
        ----------
        h_x : TYPE, optional
            DESCRIPTION. The default is 0.
        h_y : TYPE, optional
            DESCRIPTION. The default is 0.
        h_z : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        for h in [h_x, h_y, h_z]:
            if not isinstance(h, str):
                h = str(h)
                print(h)
        
        
        factor = self.Ms/2/self.kku
        #self.h_ext = factor*fen.project(fen.as_vector((h_x, h_y, h_z)), self.FS)
        
        self.h_ext = factor*fen.project(fen.Expression((h_x, h_y, h_z), degree = 4), self.FS)
    
    def set_in_cond(self, m_in_expr, boundary, in_type='new', path_to_sol = None):
        """Set initial condition and boundary condition from fenics expression"""
        
        if in_type == 'old':
            hdf_m_old = fen.HDF5File(self.mesh.mpi_comm(), path_to_sol, 'r')
            self.in_cond = fen.project(m_in_expr, self.FS)
            self.m = fen.Function(self.FS)
            self.phi_prev = fen.Function(self.FS_1d)
            hdf_m_old.read(self.m, "/m_field")
            hdf_m_old.read(self.phi_prev, "/phi_prev")
            hdf_m_old.close()
            self.BC = fen.DirichletBC(self.FS, self.m, boundary)
        elif in_type == 'new':
            self.in_cond = fen.project(m_in_expr, self.FS)
            self.m = fen.project(m_in_expr, self.FS)
            self.BC = fen.DirichletBC(self.FS, m_in_expr, boundary)
            self.phi_prev = fen.Function(self.FS_1d)
            self.phi_prev.vector()[:] = np.zeros(self.phi_prev.vector().get_local().shape)
        else:
            raise NotImplementedError()
            
        
        self.T = 1E-10
        
        m1, m2, m3 = fen.split(self.m)
        
        self.Pol = fen.project(self.m*m1.dx(0) - m1*self.m.dx(0), self.FS)
        
    def h_rest(self, m):
        """
        Construct h_rest - part of effective field, which is contained in 
        variational form

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.
        pin : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        ku_func = self.ku_func
        
        m1, m2, m3 = fen.split(m) 
        e1, e2, e3 = fen.split(self.e_v)
        dedz_1, dedz_2, dedz_3 = fen.split(self.dedz_v)
        dedy_1, dedy_2, dedy_3 = fen.split(self.dedy_v)
        
        vec = fen.as_vector((-self.p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*e1.dx(1) + m3*dedz_1), \
                             -self.p*(2*e1*m1.dx(1) + 2*e2*m2.dx(1) + 2*e3*m3.dx(1) + m1*e1.dx(1) + m2*e2.dx(1) + m3*e3.dx(1) + m1*e2.dx(0) + m2*e2.dx(1) + m3*dedz_2), \
                                  -self.p*(m1*e3.dx(0) + m2*e3.dx(1) + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3)))
        oo = fen.Constant(0)
        m1 = variable(m1)
        m2 = variable(m2)
        m3 = variable(m3)
        #mm = fen.as_vector((m1, m2, m3))
        w_an = -self.kku*m3**2*ku_func
        an_vec = fen.as_vector((-diff(w_an,m1)/2/self.kku, -diff(w_an,m2)/2/self.kku, -diff(w_an,m3)/2/self.kku))
        
        factor = self.Ms/2/self.kku
        
        #self.demag_pot()
        
        phi_vec = -fen.as_vector((self.pot.dx(0), self.pot.dx(1), oo))
        
        h_total = an_vec + vec + factor*phi_vec
        
        if self.h_ext is not None:
            h_total += self.h_ext
        
        return h_total
    
    def h_rest_0(self, m):
        """
        Construct h_rest - part of effective field, which is contained in 
        variational form

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        m1, m2, m3 = fen.split(m) 
        # e1, e2, e3 = fen.split(self.e_v)
        # dedz_1, dedz_2, dedz_3 = fen.split(self.dedz_v)
        # dedy_1, dedy_2, dedy_3 = fen.split(self.dedy_v)
        
        # vec = fen.as_vector((-self.p*(2*e1*m1.dx(0) + 2*e2*m2.dx(0) + 2*e3*m3.dx(0) + m1*e1.dx(0) + m2*e2.dx(0) + m3*e3.dx(0) + m1*e1.dx(0) + m2*dedy_1 + m3*dedz_1), \
        #              -self.p*(m1*dedy_1+ m2*dedy_2 + m3*dedy_3 + m1*e2.dx(0) + m2*dedy_2 + m3*dedz_2), \
        #                   -self.p*(m1*e3.dx(0) + m2*dedy_3 + m3*dedz_3 + m1*dedz_1 + m2*dedz_2 + m3*dedz_3)))
        #oo = fen.Constant(0)
        m1 = variable(m1)
        m2 = variable(m2)
        m3 = variable(m3)
        #mm = fen.as_vector((m1, m2, m3))
        w_an = -self.kku*m3**2
        an_vec = fen.as_vector((-diff(w_an,m1)/2/self.kku, -diff(w_an,m2)/2/self.kku, -diff(w_an,m3)/2/self.kku))
        return an_vec #+ vec + self.h_ext
    
    def dot_v(self, m, mm):
        """
        

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.
        mm : TYPE
            DESCRIPTION.

        Returns
        -------
        expr : TYPE
            DESCRIPTION.

        """
        
        mm1, mm2, mm3 = fen.split(m)
        e1, e2, e3 = fen.split(self.e_v)
        
        mm_2d = fen.as_vector((mm1, mm2))
        
        expr = fen.dot(fen.grad(fen.cross(self.w,m)[0]), fen.grad(mm1) + 2*self.p*e1*mm_2d) + \
        fen.dot(fen.grad(fen.cross(self.w,m)[1]), fen.grad(mm2) + 2*self.p*e2*mm_2d) + \
            fen.dot(fen.grad(fen.cross(self.w,m)[2]), fen.grad(mm3) + 2*self.p*e3*mm_2d)
        
        return expr
    
    def dot_v_0(self, m, mm):
        """
        

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.
        mm : TYPE
            DESCRIPTION.

        Returns
        -------
        expr : TYPE
            DESCRIPTION.

        """
        
        mm1, mm2, mm3 = fen.split(m)
        #e1, e2, e3 = fen.split(self.e_v)
        
        expr = fen.dot(fen.cross(self.w,m)[0].dx(0), mm1.dx(0)) + \
        fen.dot(fen.cross(self.w,m)[1].dx(0), mm2.dx(0)) + \
            fen.dot(fen.cross(self.w,m)[2].dx(0), mm3.dx(0))
        
        return expr
    
    def set_comp_params(self, dt, N_f, route_0 = '/media/mnv/T7/1d_pin/initial/'):
        """
        

        Parameters
        ----------
        dt : TYPE
            DESCRIPTION.
        N_f : TYPE
            DESCRIPTION.
        route_0 : TYPE, optional
            DESCRIPTION. The default is '/media/mnv/T7/resuls_oop/'.

        Returns
        -------
        None.

        """
        
        self.dt = dt
        self.N_f = N_f
        self.diffr = fen.Function(self.FS)
        
        self.route_0 = route_0
        
        self.time_txt = np.zeros(N_f)
        
        self.delta_txt = np.zeros(N_f)
        self.w_ex_txt = np.zeros(N_f)
        self.w_a_txt = np.zeros(N_f)
        self.w_hd_txt = np.zeros(N_f)
        self.w_hext_txt = np.zeros(N_f)
        self.Pol1_txt = np.zeros(N_f)
        self.Pol2_txt = np.zeros(N_f)
        self.Pol3_txt = np.zeros(N_f)
        self.w_me_txt = np.zeros(N_f)
        
        self.w_tot_txt = np.zeros(N_f)
    
    def set_F(self):
        
        f_pi = np.pi*4
        F_Pi = fen.Constant(f_pi)
        
        n = fen.FacetNormal(mesh)
        
        m1, m2, m3 = fen.split(self.m)
        m_2d = fen.as_vector((m1,m2))
        
        mb1, mb2, mb3 = fen.split(self.in_cond)
        
        m_b = fen.as_vector((mb1, mb2))
        
        Fp =  - fen.dot(fen.grad(self.wp), fen.grad(self.vp))*fen.dx + F_Pi*fen.dot(m_2d, fen.grad(self.wp))*fen.dx -F_Pi*self.wp*fen.dot(m_b,n)*fen.ds
        
        a = fen.lhs(Fp)
        L = fen.rhs(Fp)
    
        self.A_p = fen.assemble(a)
        self.b_p = fen.assemble(L)
        #BC.apply(A,b)
        self.solver_p = fen.KrylovSolver('gmres', 'hypre_euclid') #KrylovSolver('gmres', 'ilu') #hypre_euclid
        self.solver_p.parameters["nonzero_initial_guess"] = True
        
        Dt = fen.Constant(self.dt)
        self.F = fen.dot(self.w,(self.v-self.m)-self.alpha*fen.cross(self.v,(self.v-self.m)))*fen.dx + Dt*fen.dot(self.w, fen.cross(self.v, self.h_rest(self.v)))*fen.dx - Dt*self.dot_v(self.v, self.v)*fen.dx #+ dot(w,cross(m,dmdn(m,n)))*ds + 2*pp*dot(w,cross(m,e_f))*dot(to_2d(m),n)*ds
        self.Jac = fen.derivative(self.F, self.v)
    
    def solve_step(self, i):
        self.demag_pot()
        fen.solve(self.F==0, self.v, J=self.Jac, solver_parameters = { "newton_solver": { "absolute_tolerance": 1e-12, "relative_tolerance": 1e-11}})
        # self.BC
        print_0("Solution (non-normalized) max norm: ", self.max_norm(self.v))
        
        # v1, v2, v3 = fen.split(self.v)
        # fen.plot(v3)
        
        self.v.vector()[:] = self.norm_sol_s(self.v).vector()[:]
        
        print_0("Solution (normalized) max norm: ", self.max_norm(self.v))
        
        V = self.v.vector()
        M = self.m.vector()
        Diffr = V - M
        self.diffr.vector()[:] = Diffr/(self.Lx*self.Ly*self.dt)
        
        error = (self.m-self.v)**2*fen.dx
        E = fen.sqrt(abs(fen.assemble(error)))/(self.Lx*self.Ly*self.dt)
        self.delta_txt[i] = E
        print_0('delta = ', E, ', ', 'i = ', i)
        
        diff_max = np.max(np.abs(self.diffr.vector().get_local()))
        print_0('diff max = ', diff_max)
        
        self.time_txt[i] = self.T
        
        m1, m2, m3 = fen.split(self.m)
        
        w_ex = fen.assemble((fen.dot(fen.grad(m1), fen.grad(m1)) + fen. dot(fen.grad(m2), fen.grad(m2)) + fen.dot(fen.grad(m3), fen.grad(m3)))*fen.dx)/(self.Lx*self.Ly)
        self.w_ex_txt[i] = w_ex
        
        w_a = fen.assemble(-self.ku_func*m3**2*fen.dx)/(self.Lx*self.Ly)
        self.w_a_txt[i] = w_a
        
        w_hd = fen.assemble((2*np.pi*self.Ms**2*m1**2)*fen.dx)/(self.kku*self.Lx*self.Ly)
        self.w_hd_txt[i] = w_hd
        
        if self.h_ext is not None:
            w_hext = fen.assemble(-fen.dot(self.h_ext,self.m)*fen.dx)/(self.kku*self.Lx*self.Ly)
        else:
            w_hext = 0
        self.w_hext_txt[i] = w_hext
        
        Pol1 = fen.assemble((m1*m1.dx(0) - m1*m1.dx(0))*fen.dx)/(self.Lx*self.Ly)
        Pol2 = fen.assemble((m2*m1.dx(0) - m1*m2.dx(0))*fen.dx)/(self.Lx*self.Ly)
        Pol3 = fen.assemble((m3*m1.dx(0) - m1*m3.dx(0))*fen.dx)/(self.Lx*self.Ly)
        
        self.Pol1_txt[i] = Pol1
        self.Pol2_txt[i] = Pol2
        self.Pol3_txt[i] = Pol3
        
        w_me = fen.assemble(fen.dot(self.e_v, self.m*m1.dx(0) - m1*self.m.dx(0))*fen.dx)/(self.kku*self.Lx*self.Ly)
        self.w_me_txt[i] = w_me*(-2*self.p)
        
        self.w_tot_txt[i] = self.w_ex_txt[i] + self.w_a_txt[i] + self.w_hd_txt[i] + self.w_hext_txt[i] + self.w_me_txt[i]
        
        self.Pol = fen.project(self.m*m1.dx(0) - m1*self.m.dx(0), self.FS)
        
        self.m.assign(self.v)
        self.phi_prev.assign(self.pot)
        
        
    def solve_2d(self, each_idx_write = 5):
        
        e_xdmf =  fen.XDMFFile(self.route_0 + 'graphs/e.xdmf')
        e_xdmf.write(self.e_v)
        e_xdmf.close()
        
        dedz_xdmf =  fen.XDMFFile(self.route_0 + 'graphs/dedz.xdmf')
        dedz_xdmf.write(self.dedz_v)
        dedz_xdmf.close()
        
        dedy_xdmf =  fen.XDMFFile(self.route_0 + 'graphs/dedy.xdmf')
        dedy_xdmf.write(self.dedy_v)
        dedy_xdmf.close()
        
        ku_func_xdmf =  fen.XDMFFile(self.route_0 + 'graphs/ku_func.xdmf')
        ku_func_xdmf.write(self.ku_func)
        ku_func_xdmf.close()
        
        m_xdmf= fen.XDMFFile(self.route_0 + "graphs/m.xdmf")
        pot_xdmf= fen.XDMFFile(self.route_0 + "graphs/pot.xdmf")
        derivative_xdmf = fen.XDMFFile(self.route_0 + "graphs/deriv.xdmf")
        P_xdmf = fen.XDMFFile(self.route_0 + "graphs/P.xdmf")
        i = 0
        for i in range(self.N_f):
            self.solve_step(i)
            
            if (i%each_idx_write == 0):
                m_xdmf.write(self.m, self.T)
                pot_xdmf.write(self.pot, self.T)
                derivative_xdmf.write(self.diffr, self.T)
                P_xdmf.write(self.Pol, self.T)
            
            self.T += self.dt
            
        m_xdmf.close()
        pot_xdmf.close()
        derivative_xdmf.close()
        P_xdmf.close()
        
        if rank==0:
            out_txt = np.vstack((self.time_txt,
                                 self.delta_txt,
                                 self.w_ex_txt, 
                                 self.w_a_txt, 
                                 self.w_hd_txt,
                                 self.w_hext_txt,
                                 self.w_me_txt,
                                 self.w_tot_txt,
                                 self.Pol1_txt,
                                 self.Pol2_txt,
                                 self.Pol3_txt
                                 )).T
            
            out_txt_header = 't, delta, w_ex, w_a, w_hd, w_hext, w_me, w_tot, Px, Py, Pz'
            
            np.savetxt(self.route_0 + "avg_table.txt", out_txt, fmt='%.10e', delimiter=', ', header=out_txt_header, comments='')
        
        hdf_m = fen.HDF5File(self.mesh.mpi_comm(), self.route_0 + 'series_new/m_final.h5', 'w')
        hdf_m.write(self.mesh, "/my_mesh")
        hdf_m.write(self.m, "/m_field")
        hdf_m.write(self.phi_prev, "/phi_prev")
        hdf_m.close()
                
    
##########################################################################

def boundary(x, on_boundary):
    return on_boundary

Lx = 30
Ly = 30

Nx = 7
Ny = 7

mesh = fen.RectangleMesh(fen.Point(-Lx/2,-Ly/2), fen.Point(Lx/2,Ly/2), int(Nx*Lx), int(Ny*Ly))

solver_1d = llg2_solver(Lx, Ly, mesh)
solver_1d.set_params(alpha = 1E-4, pin=False)

ub = fen.Expression(("sin(2*atan(exp(x[1]/d)))", "0", "cos(2*atan(exp(x[1]/d)))"), degree = 4, d=1)

sin_0 = -4*np.pi*4*4/2/1000/(1+2*np.pi*4**2/1000)
cos_0 = np.sqrt(1-sin_0**2)

#ub = fen.Expression(("-sqrt(1-mul*mul*cos(2*atan(exp(x[0]/d)))*cos(2*atan(exp(x[0]/d))))", "0", "mul*cos(2*atan(exp(x[0]/d)))"), degree = 4, d=1, mul = cos_0)

#ub = fen.Expression(("sin_0 - a*exp(-x[0]*x[0])", "sqrt(1-pow(sin_0 - a*exp(-x[0]*x[0]), 2) - pow(cos_0*cos(2*atan(exp(x[0]/d))), 2))", "cos_0*cos(2*atan(exp(x[0]/d)))"), degree = 4, d=1, cos_0 = cos_0, sin_0 = sin_0, a = 0.7)

solver_1d.set_in_cond(ub, boundary, in_type='new', path_to_sol='/media/mnv/T7/1d_pin/initial_pin_hz_R_e_R_ER/results/series_new/m_final.h5')


solver_1d.e_field_from_ps(yy0=2, rr0 = 0.00001, UU0=3*2*10/3/100)

#solver_1d.set_h_ext(0, 0, 0) #'-0.01*1000/4*tanh(x[0]+1)'

solver_1d.set_comp_params(dt = 0.001, N_f = 100, route_0 = '/media/mnv/T7/sw_new/test/')

solver_1d.set_F()

#solver_1d.solve_step()

solver_1d.solve_2d(each_idx_write = 1)