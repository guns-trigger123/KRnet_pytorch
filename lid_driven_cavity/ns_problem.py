import torch
from abc import ABC, abstractmethod
from data_read import read_lid_driven_data


class NavierStokes2D(ABC):
    @abstractmethod
    def velocity_norm_real(self):
        pass

    @abstractmethod
    def u_velocity_real(self):
        pass

    @abstractmethod
    def v_velocity_real(self):
        pass

    @abstractmethod
    def p_velocity_real(self):
        pass


class LidDrive2D(NavierStokes2D):
    def __init__(self, Re=100):
        self.Re = Re
        self.data_column_names = [
            'x', 'y', f'spf.U @ Re={Re}', f'u @ Re={Re}', f'v @ Re={Re}', f'p @ Re={Re}',
        ]
        self.data = read_lid_driven_data(self.data_column_names)

    def velocity_norm_real(self):
        return self.data[['x', 'y', f'spf.U @ Re={self.Re}']]

    def u_velocity_real(self):
        return self.data[['x', 'y', f'u @ Re={self.Re}']]

    def v_velocity_real(self):
        return self.data[['x', 'y', f'v @ Re={self.Re}']]

    def p_velocity_real(self):
        return self.data[['x', 'y', f'p @ Re={self.Re}']]
