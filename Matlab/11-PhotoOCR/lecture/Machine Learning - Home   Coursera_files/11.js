(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{"6nMu":function(module,exports,e){"use strict";var t=e("TqRt");Object.defineProperty(exports,"__esModule",{value:!0}),exports.BaseComp=exports.default=void 0;var n=t(e("MVZn")),a=t(e("VbXa")),u=t(e("sbe7")),r=e("MnCE"),l=t(e("Puqe")),o=t(e("P5U6")),d=t(e("+LJP")),c=t(e("b+bd")),i=t(e("tODj")),s=t(e("6uhC")),f=function(e){function FullscreenLayoutHandler(){return e.apply(this,arguments)||this}(0,a.default)(FullscreenLayoutHandler,e);var t=FullscreenLayoutHandler.prototype;return t.componentDidMount=function componentDidMount(){o.default.setMarkOnce("FullscreenLayoutHandlerMounted",!0)},t.render=function render(){var e=this.props,t=e.computedItem,a=e.children;if(!t)return u.default.createElement(s.default,null);return u.default.createElement("div",{className:"rc-FullscreenLayoutHandler"},a&&u.default.cloneElement(a,(0,n.default)({},(0,l.default)(this.props,["children"]),{key:t.id})))},FullscreenLayoutHandler}(u.default.Component),p=(0,r.compose)((0,d.default)(function(e){return{itemId:e.params.item_id}}),(0,c.default)(["CourseStore"],function(e,t){var n=e.CourseStore,a=t.itemId;return{itemMetadata:n.getMaterials().getItemMetadata(a)}}),i.default)(f);exports.default=p;var m=f;exports.BaseComp=m}}]);
//# sourceMappingURL=11.9d9b36a77d380c8587b3.js.map