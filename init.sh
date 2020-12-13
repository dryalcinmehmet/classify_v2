#!/bin/bash
flask db init || echo ""  &&
flask db migrate || echo "" &&
flask db upgrade