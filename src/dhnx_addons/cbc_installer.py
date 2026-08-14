import os
import subprocess
import zipfile
import urllib.request
import requests
from tkinter import filedialog
from tkinter import Tk
import platform
import logging


class CBCInstaller:
    """Class for handling the installation of the CBC optimization software."""

    def __init__(self, gui_folder_selection=False):
        """Initialize the CBCInstaller.

        Parameters:
            gui_folder_selection (bool):
                If True, use a GUI for folder selection.  Otherwise, use
                command line.
        """
        self.gui_folder_selection = gui_folder_selection
        self.folder = os.path.join(os.path.expanduser('~'), 'coin-or-cbc')
        self.folder_bin = os.path.join(self.folder, 'bin')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(handler)

    def is_solver_installed(self, solver='cbc'):
        """Check if CBC is already installed.

        Return:
            True if CBC is installed, False otherwise.
        """
        args = {'cbc': ["cbc", "-?"],
                "gurobi": ["gurobi", "--help"],
                }
        try:
            if solver == 'cbc':
                subprocess.check_output(args[solver], shell=False)
            elif solver == 'gurobi':
                subprocess.check_output(args[solver], shell=True)
            else:
                raise ValueError(f"Solver '{solver}' not defined")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def download_and_extract_cbc(self):
        """Download and extract CBC solver from the latest release on GitHub.

        Return:
            The folder where the binaries are extracted.
        """
        # Get the latest release information from the GitHub API
        response = requests.get(
            'https://api.github.com/repos/coin-or/Cbc/releases/latest')
        response_json = response.json()

        # Get the download URL for the CBC binary for Windows
        assets = response_json['assets']
        for asset in assets:
            if self._asset_match(asset['name']):
                url = asset['browser_download_url']
                break

        filename = url.split('/')[-1]

        # Let the user choose the download location
        if self.gui_folder_selection:
            root = Tk()
            root.withdraw()  # Hide the main window
            message = ("Choose installation location or cancel to use "
                       f"default folder {self.folder}")
            self.logger.info(message)
            folder = filedialog.askdirectory(
                initialdir=os.path.expanduser('~'),
                title=message,
            )
            if folder:
                self.folder = folder
        else:
            confirmation = input("Do you want to install to the default "
                                 f"location {self.folder}? (y/n): ")
            if confirmation.lower() != 'y':
                self.folder = input("Enter the installation folder: ")

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.logger.info(f"Downloading {filename} from {url} to {self.folder}")
        urllib.request.urlretrieve(url, os.path.join(self.folder, filename))

        with zipfile.ZipFile(os.path.join(self.folder, filename),
                             'r') as zip_ref:
            zip_ref.extractall(self.folder)
        self.logger.info(f"Extracted {filename} to {self.folder}")

        os.remove(os.path.join(self.folder, filename))
        self.logger.info(f"Removed {filename}")

        return self.folder

    def add_to_path(self):
        """Add the CBC binaries to the PATH environment variable.

        This is only valid for the current runtime.
        """
        if os.path.exists(self.folder_bin):
            os.environ['PATH'] += os.pathsep + self.folder_bin
            self.logger.info(f"Added {self.folder_bin} to PATH")

    def _asset_match(self, asset_name):
        """Check if the asset name matches the current operating system.

        Parameters:
            asset_name (str): The name of the asset.

        Return:
            True if the asset matches the operating system, False otherwise.
        """
        system = platform.system()
        if system == 'Windows' and 'w64-msvc17-md.zip' in asset_name:
            return True
        elif system == 'Linux':
            raise NotImplementedError(
                f"No automatic CBC installation for {system} defined. "
                "Install with 'sudo apt install coinor-cbc -y'")
        elif system == 'Darwin':
            raise NotImplementedError(
                f"No automatic CBC installation for {system} defined")
        return False

    def install(self):
        """Install CBC if it is not installed."""
        if not self.is_solver_installed():
            self.add_to_path()
        else:
            self.logger.info('CBC is already installed.')
            return

        if not self.is_solver_installed():
            self.logger.info('CBC is not installed. Installing...')
            self.download_and_extract_cbc()
            self.add_to_path()
        else:
            self.logger.info('CBC is already installed.')


if __name__ == '__main__':
    installer = CBCInstaller()
    installer.install()
