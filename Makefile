clean:
	@find . -iname \*~ -exec rm -rfv {} \;
	@find . -name channels.avi -exec rm -rfv {} \;
