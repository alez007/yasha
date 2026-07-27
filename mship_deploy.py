"""Back-compat shim for the dev container / scripts/entrypoint.sh."""

from modelship.driver import main

if __name__ == "__main__":
    main()
