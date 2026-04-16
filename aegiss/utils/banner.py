"""Banner."""

from __future__ import annotations

BANNER = r"""
                    ######
                 ####    ####     ######
              ####          #######   ###
             ###              ###      ##
           ###               ######    ##
          ###              ####  ###   #
         ###              ###     ###
        ###         #    ###       ###
       ###          ##  ###         ###
       ###           #####          ###
      ###            ###             ###
     ###            #####             ###
     ###           ###  ##            ###
    ###          ####    ##            ###
    ###         ###       ###          ###
    ###        ###         ###         ###
    ##       ####            ###        ##
    ##      ###               ####      ##
    ##    ####                  ####    ##
    ########                      ########
     ####                            ###

  AEGISS  —  Atomic Orbital and Entropy based Guided Inference for Space Selection
"""


def print_banner() -> None:
    """Print the AEGISS ASCII art banner to stdout."""
    print(BANNER)
